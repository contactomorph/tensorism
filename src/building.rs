use std::fmt::Debug;
use std::marker::{Copy, PhantomData};
use crate::dimensions::*;
use crate::tensors::*;
use crate::shapes::{Shape, Tensor1Shape, Tensor2Shape, Tensor3Shape};

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
