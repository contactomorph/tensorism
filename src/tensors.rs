//! The mathematical word tensor is used here more broadly to describe
//! arrays with multiple indexes containing copiable data.

use crate::shapes::{Shape, Tensor1Shape, Tensor2Shape};

use super::dimensions::*;
use std::fmt::{Debug, Error, Formatter};
use std::marker::{Copy, PhantomData};

/// A general trait shared by all tensor types.
pub trait Tensor: Eq {
    /// The type of elements stored in the tensor.
    type Element: Eq + Debug;
    /// The type of the dimensions tuple
    type Dimensions: Copy + Eq + Debug;
    /// The number of dimensions, also corresponding
    /// to the number of indexes.
    const RANK: u16;
    /// The total number of coordinates
    fn count(&self) -> u64;
    /// The tuple of dimensions
    fn dims(&self) -> Self::Dimensions;
}

#[derive(PartialEq, Eq)]
pub struct Tensor0<V: Eq + Debug> {
    data: V,
}

#[derive(PartialEq, Eq)]
pub struct Tensor1<T: DimTag, V: Eq + Debug> {
    phantom: PhantomData<T>,
    data: Vec<V>,
}

#[derive(PartialEq, Eq)]
pub struct Tensor2<T1: DimTag, T2: DimTag, V: Eq + Debug> {
    d1: Dim<T1>,
    d2: Dim<T2>,
    data: Vec<V>,
}

#[derive(PartialEq, Eq)]
pub struct Tensor3<T1: DimTag, T2: DimTag, T3: DimTag, V: Eq + Debug> {
    phantom: PhantomData<(T1, T2, T3)>,
    #[allow(dead_code)]
    data: Vec<V>,
}

#[derive(PartialEq, Eq)]
pub struct Tensor4<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, V: Eq + Debug> {
    phantom: PhantomData<(T1, T2, T3, T4)>,
    #[allow(dead_code)]
    data: Vec<V>,
}

pub type Scalar<V> = Tensor0<V>;
pub type Vector<T, V> = Tensor1<T, V>;
pub type Matrix<T1, T2, V> = Tensor2<T1, T2, V>;

pub type StaticVector<const N: usize, V> = Tensor1<StaticDimTag<N>, V>;
pub type StaticMatrix<const N1: usize, const N2: usize, V> =
    Tensor2<StaticDimTag<N1>, StaticDimTag<N2>, V>;

impl<V: Eq + Debug> Tensor for Tensor0<V> {
    type Element = V;
    type Dimensions = ();
    const RANK: u16 = 0;
    fn count(&self) -> u64 {
        1
    }
    fn dims(&self) -> Self::Dimensions {}
}

impl<T: DimTag, V: Eq + Debug> Tensor for Tensor1<T, V> {
    type Element = V;
    type Dimensions = (Dim<T>,);
    const RANK: u16 = 1;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn dims(&self) -> Self::Dimensions {
        let n = self.data.len();
        unsafe { (Dim::<T>::unsafe_new(n),) }
    }
}

impl<T1: DimTag, T2: DimTag, V: Eq + Debug> Tensor for Tensor2<T1, T2, V> {
    type Element = V;
    type Dimensions = (Dim<T1>, Dim<T2>);
    const RANK: u16 = 2;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn dims(&self) -> Self::Dimensions {
        (self.d1, self.d2)
    }
}

impl<V: Clone + Eq + Debug> Clone for Tensor0<V> {
    fn clone(&self) -> Self {
        Tensor0 {
            data: self.data.clone(),
        }
    }
}

impl<V: Copy + Eq + Debug> Copy for Tensor0<V> {}

impl<T: DimTag, V: Clone + Eq + Debug> Clone for Tensor1<T, V> {
    fn clone(&self) -> Self {
        Tensor1 {
            phantom: PhantomData {},
            data: self.data.clone(),
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: Clone + Eq + Debug> Clone for Tensor2<T1, T2, V> {
    fn clone(&self) -> Self {
        Tensor2 {
            d1: self.d1,
            d2: self.d2,
            data: self.data.clone(),
        }
    }
}

impl<V: Eq + Debug> Debug for Tensor0<V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}\u{3009}[")?;
        self.data.fmt(formatter)?;
        formatter.write_str("]")
    }
}

impl<T: DimTag, V: Eq + Debug> Debug for Tensor1<T, V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        let d = unsafe { Dim::<T>::unsafe_new(self.data.len()) };
        formatter.write_str("\u{3008}")?;
        d.fmt(formatter)?;
        formatter.write_str("\u{3009}[")?;
        let mut first = true;
        for elt in self.data.iter().take(10) {
            if first {
                formatter.write_str(", ")?
            }
            elt.fmt(formatter)?;
            first = false;
        }
        if self.data.len() <= 10 {
            formatter.write_str("]")
        } else {
            formatter.write_str(", \u{2026}]")
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: Eq + Debug> Debug for Tensor2<T1, T2, V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str("\u{3009}[")?;
        let mut i = 0usize;
        for elt in self.data.iter().take(10) {
            if i > 0 {
                if i % self.d2.as_usize() == 0 {
                    formatter.write_str(" | ")?
                } else {
                    formatter.write_str(", ")?
                }
            }
            elt.fmt(formatter)?;
            i += 1
        }
        if self.data.len() <= 10 {
            formatter.write_str("]")
        } else {
            formatter.write_str(", \u{2026}]")
        }
    }
}

impl<V: Eq + Debug> Tensor0<V> {
    pub fn new(value: V) -> Self {
        Tensor0::<V> { data: value }
    }
    pub fn get(&self) -> &V {
        &self.data
    }
}

impl<T: DimTag, V: Eq + Debug> Tensor1<T, V> {
    pub fn get(&self, i: usize) -> &V {
        &self.data[i]
    }

    pub unsafe fn get_unchecked(&self, i: usize) -> &V {
        self.data.get_unchecked(i)
    }

    pub fn set(&mut self, i: usize, value: V) {
        self.data[i] = value
    }

    pub unsafe fn set_unchecked(&mut self, _i: usize, _value: V) {
        //self.data.get_unchecked_mut(index) = value
        panic!("")
    }

    pub fn from_vector_with_dim(d: Dim<T>, mut values: Vec<V>) -> Self {
        if values.len() < d.as_usize() {
            panic!("Array too short")
        }
        values.truncate(d.as_usize());
        Tensor1::<T, V> {
            phantom: PhantomData {},
            data: values,
        }
    }
    pub fn set___<VInput: Eq + Debug>(
        &mut self,
        input: &Tensor1<T, VInput>,
        mut f: impl FnMut(&VInput) -> V,
    ) {
        let ptr_o = self.data.as_mut_ptr();
        let ptr = input.data.as_ptr();
        for i in 0usize..self.data.len() {
            unsafe {
                let ptr = ptr.add(i);
                *ptr_o.add(i) = f(&*ptr);
            }
        }
    }
    pub fn set_from_2<V1: Eq + Debug, V2: Eq + Debug>(
        &mut self,
        input1: &Tensor1<T, V1>,
        input2: &Tensor1<T, V2>,
        mut f: impl FnMut(&V1, &V2) -> V,
    ) {
        let ptr_o = self.data.as_mut_ptr();
        let ptr1 = input1.data.as_ptr();
        let ptr2 = input2.data.as_ptr();
        for i in 0usize..self.data.len() {
            unsafe {
                let ptr1 = ptr1.add(i);
                let ptr2 = ptr2.add(i);
                *ptr_o.add(i) = f(&*ptr1, &*ptr2);
            }
        }
    }

    pub fn map<W: Eq + Debug>(&self, mut _f: impl FnMut(&V) -> W) -> Tensor1<T, W> {
        panic!("not impl")
    }
}

impl<T: DimTag, V: Eq + Debug> Tensor1<T, V> {
    pub fn try_cast<DPrime: DimTag>(self, d: Dim<DPrime>) -> Result<Tensor1<DPrime, V>, Self> {
        if self.data.len() == d.as_usize() {
            Ok(Tensor1::<DPrime, V> {
                phantom: PhantomData {},
                data: self.data,
            })
        } else {
            Err(self)
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: Eq + Debug> Tensor2<T1, T2, V> {
    pub unsafe fn get(&self, i1: usize, i2: usize) -> &V {
        if i1 >= self.d1.as_usize() || i2 >= self.d2.as_usize() {
            panic!("Incorrect index")
        }
        self.get_unchecked(i1, i2)
    }

    pub unsafe fn get_unchecked(&self, i1: usize, i2: usize) -> &V {
        self.data.get_unchecked(i1 * self.d2.as_usize() + i2)
    }

    pub fn try_cast<D1Prime: DimTag, D2Prime: DimTag>(
        self,
        d1: Dim<D1Prime>,
        d2: Dim<D2Prime>,
    ) -> Result<Tensor2<D1Prime, D2Prime, V>, Self> {
        if self.d1 == d1 && self.d2 == d2 {
            Ok(Tensor2::<D1Prime, D2Prime, V> {
                d1,
                d2,
                data: self.data,
            })
        } else {
            Err(self)
        }
    }
}

#[doc(hidden)]
pub unsafe fn from_array_to_tensor1_unchecked<T: DimTag, V: Copy + Eq + Debug>(
    values: Vec<V>,
) -> Tensor1<T, V> {
    Tensor1::<T, V> {
        phantom: PhantomData {},
        data: values,
    }
}

pub fn from_array<V: Copy + Eq + Debug, const N: usize>(
    array: &[V; N],
) -> Tensor1<StaticDimTag<N>, V> {
    Tensor1::<StaticDimTag<N>, V> {
        phantom: PhantomData {},
        data: array.to_vec(),
    }
}

#[macro_export]
macro_rules! from_array_to_tensor1 {
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
        unsafe { from_array_to_tensor1_unchecked::<UnnamedTag, _>($values) }
    }};
}

pub trait TensorBuilder<V> {
    type Tensor: Tensor;
    type Indices;
    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone;
    fn define(&self, f: impl FnMut(Self::Indices) -> V) -> Self::Tensor;
}

impl<T: DimTag, V: Eq + Debug> TensorBuilder<V> for Tensor1Shape<T> {
    type Tensor = Tensor1<T, V>;
    type Indices = usize;

    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone,
    {
        Self::Tensor {
            phantom: PhantomData,
            data: vec![value.clone(); self.count()],
        }
    }
    fn define(&self, mut f: impl FnMut(Self::Indices) -> V) -> Self::Tensor {
        let mut data = Vec::<V>::with_capacity(self.d.as_usize());
        for i in 0..self.d.as_usize() {
            data.push(f(i))
        }
        Self::Tensor {
            phantom: PhantomData,
            data,
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: Eq + Debug> TensorBuilder<V> for Tensor2Shape<T1, T2> {
    type Tensor = Tensor2<T1, T2, V>;
    type Indices = (usize, usize);

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
    fn define(&self, mut f: impl FnMut(Self::Indices) -> V) -> Self::Tensor {
        let mut data = Vec::<V>::with_capacity(self.count());
        for i in 0..self.d1.as_usize() {
            for j in 0..self.d2.as_usize() {
                data.push(f((i, j)));
            }
        }
        Self::Tensor {
            d1: self.d1,
            d2: self.d2,
            data,
        }
    }
}
