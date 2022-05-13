//! The mathematical word tensor is used here more broadly to describe
//! arrays with multiple indexes containing copiable data.

use std::marker::{Copy, PhantomData};
use std::fmt::{Debug, Error, Formatter};
use super::dimensions::*;

/// A general trait shared by all tensor types.
pub trait Tensor {
    /// The type of elements stored in the tensor.
    type Element: Copy;
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

pub struct Tensor0<V: Copy> {
    data: V,
}

pub struct Tensor1<D: DimTag, V: Copy> {
    phantom: PhantomData<D>,
    data: Vec<V>,
}

pub struct Tensor2<D1: DimTag, D2: DimTag, V: Copy> {
    d1: Dim<D1>,
    d2: Dim<D2>,
    data: Vec<V>,
}

pub struct Tensor3<D1: DimTag, D2: DimTag, D3: DimTag, V: Copy> {
    phantom: PhantomData<(D1, D2, D3)>,
    data: Vec<V>,
}

pub struct Tensor4<D1: DimTag, D2: DimTag, D3: DimTag, D4: DimTag, V: Copy> {
    phantom: PhantomData<(D1, D2, D3, D4)>,
    data: Vec<V>,
}

pub type Scalar<V> = Tensor0<V>;
pub type Vector<D, V> = Tensor1<D, V>;
pub type Matrix<D1, D2, V> = Tensor2<D1, D2, V>;

impl<V: Copy> Tensor for Tensor0<V> {
    type Element = V;
    type Dimensions = ();
    const RANK: u16 = 0;
    fn count(&self) -> u64 { 1 }
    fn dims(&self) -> Self::Dimensions {}
}

impl<D: DimTag, V: Copy> Tensor for Tensor1<D, V> {
    type Element = V;
    type Dimensions = (Dim<D>,);
    const RANK: u16 = 1;
    fn count(&self) -> u64 { self.data.len() as u64 }
    fn dims(&self) -> Self::Dimensions {
        let n = self.data.len();
        unsafe { (Dim::<D>::unsafe_new(n),) }
    }
}

impl<D1: DimTag, D2: DimTag, V: Copy> Tensor for Tensor2<D1, D2, V> {
    type Element = V;
    type Dimensions = (Dim<D1>, Dim<D2>);
    const RANK: u16 = 2;
    fn count(&self) -> u64 { self.data.len() as u64 }
    fn dims(&self) -> Self::Dimensions {
        (self.d1, self.d2)
    }
}

impl<V: Copy> Clone for Tensor0<V> {
    fn clone(&self) -> Self {
        Tensor0 { data: self.data }
    }
}

impl<V: Copy> Copy for Tensor0<V> {}

impl<D: DimTag, V: Copy> Clone for Tensor1<D, V> {
    fn clone(&self) -> Self {
        Tensor1 {
            phantom: PhantomData {},
            data: self.data.clone(),
        }
    }
}

impl<D1: DimTag, D2: DimTag, V: Copy> Clone for Tensor2<D1, D2, V> {
    fn clone(&self) -> Self {
        Tensor2 {
            d1: self.d1,
            d2: self.d2,
            data: self.data.clone(),
        }
    }
}

impl<V: Copy + Debug> Debug for Tensor0<V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}\u{3009}[")?;
        self.data.fmt(formatter)?;
        formatter.write_str("]")
    }
}

impl<D: DimTag, V: Copy + Debug> Debug for Tensor1<D, V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        let d = unsafe { Dim::<D>::unsafe_new(self.data.len()) };
        formatter.write_str("\u{3008}")?;
        d.fmt(formatter)?;
        formatter.write_str("\u{3009}[")?;
        let mut i = 0usize;
        for elt in self.data.iter().take(10) {
            if i > 0 { formatter.write_str(", ")? }
            elt.fmt(formatter)?;
            i += 1
        }
        if i <= 10 {
            formatter.write_str("]")
        } else {
            formatter.write_str(", \u{2026}]")
        }
    }
}

impl<V: Copy> Tensor0<V> {
    pub fn new(value: V) -> Self {
        Tensor0::<V> { data: value, }
    }
}

impl<D: DimTag, V: Copy> Tensor1<D, V> {
    pub fn new(d: Dim<D>, value: V) -> Self {
        Tensor1::<D, V> {
            phantom: PhantomData {},
            data: vec![value; d.as_usize()],
        }
    }

    pub fn set<VInput: Copy>(
        &mut self,
        input: &Tensor1<D, VInput>,
        f: impl Fn(VInput) -> V,
    ) {
        let ptr_o = self.data.as_mut_ptr();
        let ptr = input.data.as_ptr();
        for i in 0usize..self.data.len() {
            unsafe {
                let ptr = ptr.offset(i as isize);
                *ptr_o.add(i) = f(*ptr);
            }
        }
    }

    pub fn set_from_2<D1: Copy, D2: Copy>(
        &mut self,
        input1: &Tensor1<D, D1>,
        input2: &Tensor1<D, D2>,
        f: impl Fn(D1, D2) -> V,
    ) {
        let ptr_o = self.data.as_mut_ptr();
        let ptr1 = input1.data.as_ptr();
        let ptr2 = input2.data.as_ptr();
        for i in 0usize..self.data.len() {
            unsafe {
                let ptr1 = ptr1.offset(i as isize);
                let ptr2 = ptr2.offset(i as isize);
                *ptr_o.add(i) = f(*ptr1, *ptr2);
            }
        }
    }
}

#[doc(hidden)]
pub unsafe fn unsafe_from_array_to_tensor1<D: DimTag, V: Copy>(values: Vec<V>) -> Tensor1<D, V> {
    Tensor1::<D, V> {
        phantom: PhantomData {},
        data: values,
    }
}

pub fn new_from_array<V: Copy, const N: usize>(array: &[V; N]) -> Tensor1<StaticDimTag<N>, V> {
    Tensor1::<StaticDimTag<N>, V> {
        phantom: PhantomData {},
        data: array.to_vec(),
    }
}

#[macro_export]
macro_rules! from_array_to_tensor1 {
    ($values:expr) => {{
        #[allow(bad_style)]
        enum UnnamedTag {}
        lazy_static! {
            static ref THUMBPRINT: u16 = {
                let mut rng = rand::thread_rng();
                rng.gen::<u16>()
            };
        }
        impl DimTag for UnnamedTag {
            fn get_thumbprint() -> u16 { *THUMBPRINT }
        }
        unsafe { unsafe_from_array_to_tensor1::<UnnamedTag, _>($values) }
    }}
}
