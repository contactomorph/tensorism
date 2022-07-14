use crate::dimensions::*;
use crate::tensors::*;
use std::fmt::{Debug, Error, Formatter};
use std::marker::{Copy, PhantomData};

pub trait Shape {
    /// The number of dimensions, also corresponding
    /// to the number of indexes.
    const RANK: u16;
    /// The total number of coordinates
    fn count(&self) -> usize;
}

#[derive(Clone, Copy)]
pub struct Tensor1Shape<T: DimTag> {
    pub(crate) d: Dim<T>,
    size: usize,
}

#[derive(Clone, Copy)]
pub struct Tensor2Shape<T1: DimTag, T2: DimTag> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    size: usize,
}

#[derive(Clone, Copy)]
pub struct Tensor3Shape<T1: DimTag, T2: DimTag, T3: DimTag> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    size: usize,
}

impl<T: DimTag> Shape for Tensor1Shape<T> {
    const RANK: u16 = 1;

    fn count(&self) -> usize {
        self.size
    }
}

impl<T1: DimTag, T2: DimTag> Shape for Tensor2Shape<T1, T2> {
    const RANK: u16 = 2;

    fn count(&self) -> usize {
        self.size
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag> Shape for Tensor3Shape<T1, T2, T3> {
    const RANK: u16 = 3;

    fn count(&self) -> usize {
        self.size
    }
}

impl<L: DimTag, R: DimTag> PartialEq<Tensor1Shape<R>> for Tensor1Shape<L> {
    fn eq(&self, other: &Tensor1Shape<R>) -> bool {
        self.d == other.d
    }
}

impl<T: DimTag> Eq for Tensor1Shape<T> {}

impl<L1: DimTag, L2: DimTag, R1: DimTag, R2: DimTag> PartialEq<Tensor2Shape<R1, R2>>
    for Tensor2Shape<L1, L2>
{
    fn eq(&self, other: &Tensor2Shape<R1, R2>) -> bool {
        self.d1 == other.d1 && self.d2 == other.d2
    }
}

impl<T1: DimTag, T2: DimTag> Eq for Tensor2Shape<T1, T2> {}

impl<L1: DimTag, L2: DimTag, L3: DimTag, R1: DimTag, R2: DimTag, R3: DimTag>
    PartialEq<Tensor3Shape<R1, R2, R3>> for Tensor3Shape<L1, L2, L3>
{
    fn eq(&self, other: &Tensor3Shape<R1, R2, R3>) -> bool {
        self.d1 == other.d1 && self.d2 == other.d2 && self.d3 == other.d3
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag> Eq for Tensor3Shape<T1, T2, T3> {}

impl<T: DimTag> Debug for Tensor1Shape<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

impl<T1: DimTag, T2: DimTag> Debug for Tensor2Shape<T1, T2> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag> Debug for Tensor3Shape<T1, T2, T3> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d3.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

pub enum TensorBuilding {}

impl TensorBuilding {
    pub fn with_static<const N: usize>() -> Tensor1Shape<StaticDimTag<N>> {
        Tensor1Shape {
            d: new_static_dim::<N>(),
            size: N,
        }
    }
    pub fn with<T: DimTag>(dimension: Dim<T>) -> Tensor1Shape<T> {
        Tensor1Shape {
            d: dimension,
            size: dimension.as_usize(),
        }
    }
    pub fn from_array_to_tensor1<V: Copy + PartialEq + Debug, const N: usize>(
        array: &[V; N],
    ) -> Tensor1<StaticDimTag<N>, V> {
        Tensor1::<StaticDimTag<N>, V> {
            phantom: PhantomData {},
            data: array.to_vec(),
        }
    }
}

#[doc(hidden)]
pub unsafe fn from_vec_to_tensor1_unchecked<T: DimTag, V: Copy + PartialEq + Debug>(
    values: Vec<V>,
) -> Tensor1<T, V> {
    Tensor1::<T, V> {
        phantom: PhantomData {},
        data: values,
    }
}

#[macro_export]
macro_rules! from_vec_to_tensor1 {
    ($values:expr) => {{
        enum UnnamedTag {}
        lazy_static::lazy_static! {
            static ref THUMBPRINT: u16 = generate_thumbprint();
        }
        impl DimTag for UnnamedTag {
            fn get_thumbprint() -> u16 {
                *THUMBPRINT
            }
        }
        unsafe { from_vec_to_tensor1_unchecked::<UnnamedTag, _>($values) }
    }};
}

impl<T: DimTag> Tensor1Shape<T> {
    pub fn with_static<const N: usize>(&self) -> Tensor2Shape<T, StaticDimTag<N>> {
        Tensor2Shape {
            d1: self.d,
            d2: new_static_dim::<N>(),
            size: self.size.checked_mul(N).expect("Tensor size is too big"),
        }
    }
    pub fn with<T2: DimTag>(&self, dimension: Dim<T2>) -> Tensor2Shape<T, T2> {
        Tensor2Shape {
            d1: self.d,
            d2: dimension,
            size: self
                .size
                .checked_mul(dimension.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_first(&self) -> Tensor2Shape<T, T> {
        Tensor2Shape {
            d1: self.d,
            d2: self.d,
            size: self
                .size
                .checked_mul(self.size)
                .expect("Tensor size is too big"),
        }
    }
}

impl<T1: DimTag, T2: DimTag> Tensor2Shape<T1, T2> {
    pub fn with_static<const N: usize>(&self) -> Tensor3Shape<T1, T2, StaticDimTag<N>> {
        Tensor3Shape {
            d1: self.d1,
            d2: self.d2,
            d3: new_static_dim::<N>(),
            size: self.size.checked_mul(N).expect("Tensor size is too big"),
        }
    }
    pub fn with<T3: DimTag>(&self, dimension: Dim<T3>) -> Tensor3Shape<T1, T2, T3> {
        Tensor3Shape {
            d1: self.d1,
            d2: self.d2,
            d3: dimension,
            size: self
                .size
                .checked_mul(dimension.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_first(&self) -> Tensor3Shape<T1, T2, T1> {
        Tensor3Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d1,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_second(&self) -> Tensor3Shape<T1, T2, T2> {
        Tensor3Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d2,
            size: self
                .size
                .checked_mul(self.d2.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn switch_12(&self) -> Tensor2Shape<T2, T1> {
        Tensor2Shape {
            d1: self.d2,
            d2: self.d1,
            size: self.size,
        }
    }
}

pub struct TensorPreparation<T: Tensor> {
    expected_size: usize,
    dims: T::Dimensions,
    data: Vec<T::Element>,
    generator: fn(T::Dimensions, Vec<T::Element>) -> T,
}

impl<T: Tensor> TensorPreparation<T> {
    pub fn count_set_elements(&self) -> usize {
        self.data.len()
    }
    pub fn count_unset_elements(&self) -> usize {
        self.expected_size - self.data.len()
    }
    pub fn append_vec(mut self, values: &mut Vec<T::Element>) -> Self {
        self.data.append(values);
        self.data.truncate(self.expected_size);
        self
    }
    pub fn append_array<const N: usize>(mut self, values: [T::Element; N]) -> Self {
        self.data.append(&mut Vec::<_>::from(values));
        self.data.truncate(self.expected_size);
        self
    }
    pub fn fill(mut self, value: &T::Element) -> T
    where
        T::Element: Clone,
    {
        let missing_elements_count = self.expected_size - self.data.len();
        self.data
            .append(&mut vec![value.clone(); missing_elements_count]);
        self.generate()
    }
    pub fn fill_during(mut self, value: &T::Element, additional_count: usize) -> Self
    where
        T::Element: Clone,
    {
        let missing_elements_count = (self.expected_size - self.data.len()).min(additional_count);
        self.data
            .append(&mut vec![value.clone(); missing_elements_count]);
        self
    }
    pub fn generate(self) -> T {
        if self.data.len() != self.expected_size {
            panic!("Invalid size")
        }
        (self.generator)(self.dims, self.data)
    }
    pub fn try_generate(self) -> Result<T, Self> {
        if self.data.len() != self.expected_size {
            Err(self)
        } else {
            Ok((self.generator)(self.dims, self.data))
        }
    }
}

pub trait TensorBuilder<V> {
    type Tensor: Tensor;
    type MultiIndex: Eq + Debug + Copy;
    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone;
    fn define(&self, f: impl FnMut(Self::MultiIndex) -> V) -> Self::Tensor;
    fn prepare(&self) -> TensorPreparation<Self::Tensor>;
}

impl<T: DimTag, V: PartialEq + Debug> TensorBuilder<V> for Tensor1Shape<T> {
    type Tensor = Tensor1<T, V>;
    type MultiIndex = usize;

    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone,
    {
        Self::Tensor {
            phantom: PhantomData,
            data: vec![value.clone(); self.count()],
        }
    }
    fn define(&self, mut f: impl FnMut(Self::MultiIndex) -> V) -> Self::Tensor {
        let mut data = Vec::<V>::with_capacity(self.d.as_usize());
        for i in 0..self.d.as_usize() {
            data.push(f(i))
        }
        Self::Tensor {
            phantom: PhantomData,
            data,
        }
    }
    fn prepare(&self) -> TensorPreparation<Self::Tensor> {
        TensorPreparation {
            expected_size: self.count(),
            data: Vec::<V>::with_capacity(self.count()),
            dims: (self.d,),
            generator: |_dims, data| Self::Tensor {
                phantom: PhantomData,
                data,
            },
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> TensorBuilder<V> for Tensor2Shape<T1, T2> {
    type Tensor = Tensor2<T1, T2, V>;
    type MultiIndex = (usize, usize);

    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone,
    {
        Self::Tensor {
            d1: self.d1,
            d2: self.d2,
            data: vec![value.clone(); self.count()],
        }
    }
    fn define(&self, mut f: impl FnMut(Self::MultiIndex) -> V) -> Self::Tensor {
        let mut data = Vec::<V>::with_capacity(self.count());
        for i1 in 0..self.d1.as_usize() {
            for i2 in 0..self.d2.as_usize() {
                data.push(f((i1, i2)));
            }
        }
        Self::Tensor {
            d1: self.d1,
            d2: self.d2,
            data,
        }
    }
    fn prepare(&self) -> TensorPreparation<Self::Tensor> {
        TensorPreparation {
            expected_size: self.count(),
            data: Vec::<V>::with_capacity(self.count()),
            dims: (self.d1, self.d2),
            generator: |dims, data| Self::Tensor {
                d1: dims.0,
                d2: dims.1,
                data,
            },
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> TensorBuilder<V>
    for Tensor3Shape<T1, T2, T3>
{
    type Tensor = Tensor3<T1, T2, T3, V>;
    type MultiIndex = (usize, usize, usize);

    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone,
    {
        Self::Tensor {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            data: vec![value.clone(); self.count()],
        }
    }
    fn define(&self, mut f: impl FnMut(Self::MultiIndex) -> V) -> Self::Tensor {
        let mut data = Vec::<V>::with_capacity(self.count());
        for i1 in 0..self.d1.as_usize() {
            for i2 in 0..self.d2.as_usize() {
                for i3 in 0..self.d3.as_usize() {
                    data.push(f((i1, i2, i3)));
                }
            }
        }
        Self::Tensor {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            data,
        }
    }
    fn prepare(&self) -> TensorPreparation<Self::Tensor> {
        TensorPreparation {
            expected_size: self.count(),
            data: Vec::<V>::with_capacity(self.count()),
            dims: (self.d1, self.d2, self.d3),
            generator: |dims, data| Self::Tensor {
                d1: dims.0,
                d2: dims.1,
                d3: dims.2,
                data,
            },
        }
    }
}
