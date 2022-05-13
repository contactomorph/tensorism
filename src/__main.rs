#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate base64;

use std::marker::{PhantomData, Copy};
use std::fmt::{Debug, Error, Formatter};
use rand::Rng;

pub trait Tag {
    fn get_thumbprint() -> u16;
}

pub struct StaticTag<const N: usize> {}

impl<const N: usize> Tag for StaticTag<N> {
    fn get_thumbprint() -> u16 { 0 }
}

pub struct Dim<T: Tag> {
    phantom: PhantomData<T>,
    v: usize,
}

impl<T: Tag> Dim<T> {
    #[doc(hidden)]
    fn private_new(v: usize) -> Self {
        Dim { phantom: PhantomData {}, v }
    }

    pub fn value(&self) -> usize { self.v }
}


impl<T: Tag> Debug for Dim<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        let tp = T::get_thumbprint();
        self.v.fmt(formatter)?;
        if tp != 0u16 {
            formatter.write_str("|")?;
            let b64 = base64::encode(tp.to_ne_bytes());
            formatter.write_str(b64.as_str())?;
        }
        Ok(())
    }
}

impl<T: Tag> Clone for Dim<T> {
    fn clone(&self) -> Self {
        Dim {
            phantom: PhantomData {},
            v: self.v
        }
    }
}

impl<T: Tag> Copy for Dim<T> {}

pub type StaticDim<const N: usize> = Dim<StaticTag<N>>;

macro_rules! new_dim {
    ($v:expr) => {{
        #[allow(bad_style)]
        enum UnnamedTag {}
        lazy_static! {
            static ref THUMBPRINT: u16 = {
                let mut rng = rand::thread_rng();
                rng.gen::<u16>()
            };
        }
        impl Tag for UnnamedTag {
            fn get_thumbprint() -> u16 { *THUMBPRINT }
        }
        Dim::<UnnamedTag>::private_new($v)
    }}
}

pub fn new_static_dim<const N: usize>() -> Dim<StaticTag<N>> {
    Dim { phantom: PhantomData {}, v: N }
}

pub trait Tensor {
    fn dim_count() -> u16;
    fn count(&self) -> u64;
}

pub struct Tensor0<D: Copy> {
    data: D,
}

impl<D: Copy> Tensor0<D> {
    pub fn new(value: D) -> Self {
        Tensor0::<D> {data: value, }
    }

    pub fn dims(&self) -> () {}
}

impl<D: Copy> Clone for Tensor0<D> {
    fn clone(&self) -> Self {
        Tensor0::<D> { data: self.data }
    }
}

impl<D: Copy + Debug> Debug for Tensor0<D> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}>[")?;
        self.data.fmt(formatter)?;
        formatter.write_str("]")
    }
}

pub struct Tensor1<T: Tag, D: Copy> {
    phantom: PhantomData<T>,
    data: Vec<D>,
}

impl<T: Tag, D: Copy> Tensor1<T, D> {
    pub fn new(d: Dim<T>, value: D) -> Self {
        Tensor1::<T, D> {
            phantom: PhantomData {},
            data: vec![value; d.value()],
        }
    }

    pub fn dims(&self) -> (Dim<T>,) {
        let n = self.data.len();
        (Dim::<T>::private_new(n),)
    }

    pub fn set<DInput: Copy>(
        &mut self,
        input: &Tensor1<T, DInput>,
        f: impl Fn(DInput) -> D,
    ) {
        let o_ptr = self.data.as_mut_ptr();
        let i_ptr = input.data.as_ptr();
        for i in 0usize..self.data.len() {
            unsafe {
                let ptr = i_ptr.offset(i as isize);
                *o_ptr.add(i) = f(*ptr);
            }
        }
    }

    pub fn set_from_2<D1: Copy, D2: Copy>(
        &mut self,
        input1: &Tensor1<T, D1>,
        input2: &Tensor1<T, D2>,
        f: impl Fn(D1, D2) -> D,
    ) {
        let o_ptr = self.data.as_mut_ptr();
        let i1_ptr = input1.data.as_ptr();
        let i2_ptr = input2.data.as_ptr();
        for i in 0usize..self.data.len() {
            unsafe {
                let ptr1 = i1_ptr.offset(i as isize);
                let ptr2 = i2_ptr.offset(i as isize);
                *o_ptr.add(i) = f(*ptr1, *ptr2);
            }
        }
    }
}

impl<T: Tag, D: Copy> Clone for Tensor1<T, D> {
    fn clone(&self) -> Self {
        Tensor1::<T, D> {
            phantom: PhantomData {},
            data: self.data.clone(),
        }
    }
}

impl<T: Tag, D: Copy + Debug> Debug for Tensor1<T, D> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        let d = Dim::<T>::private_new(self.data.len());
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

#[doc(hidden)]
pub unsafe fn unsafe_from_array_to_tensor1<T: Tag, D: Copy>(values: Vec<D>) -> Tensor1<T, D> {
    Tensor1::<T, D> {
        phantom: PhantomData {},
        data: values,
    }
}

pub fn new_from_array<D: Copy, const N: usize>(array: &[D; N]) -> Tensor1<StaticTag<N>, D> {
    Tensor1::<StaticTag<N>, D> {
        phantom: PhantomData {},
        data: array.to_vec(),
    }
}

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
        impl Tag for UnnamedTag {
            fn get_thumbprint() -> u16 { *THUMBPRINT }
        }
        unsafe { unsafe_from_array_to_tensor1::<UnnamedTag, _>($values) }
    }}
}

pub struct Tensor2<T1: Tag, T2: Tag, D: Copy> {
    d1: Dim<T1>,
    d2: Dim<T2>,
    data: Vec<D>,
}

impl<T1: Tag, T2: Tag, D: Copy> Tensor2<T1, T2, D> {
    pub fn new(d1: Dim<T1>, d2: Dim<T2>, value: D) -> Self {
        let d = d1.value() * d2.value();
        Tensor2::<T1, T2, D> { d1, d2, data: vec![value; d] }
    }

    pub fn dims(&self) -> (Dim<T1>, Dim<T2>) { (self.d1, self.d2) }
}


impl<T1: Tag, T2: Tag, D: Copy + Debug> Debug for Tensor2<T1, T2, D> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str("\u{3009}[")?;
        let mut i = 0usize;
        for elt in self.data.iter().take(10) {
            if i > 0 { formatter.write_str(", ")? }
            elt.fmt(formatter)?;
            i += 1
        }
        formatter.write_str(", \u{2026}]")
    }
}

fn main() {
    let n = 3 + 2;
    let d = new_dim!(n);
    let static_d = new_static_dim::<7>();
    println!("Dimensions, dynamic:{:?}, static:{:?}", d, static_d);
    let mut t = Tensor1::new(d, 'a');
    let a = Tensor2::new(d, d, 1);
    let v = Tensor1::new(d, 23.4);
    t.set(&v, |elt| if elt < 0.0 {'x'} else { 'y' });
    let b = from_array_to_tensor1!(vec!["Hello", "World", "How are you"]);
    println!("Tensor t: {:?}", t);
    println!("Tensor a: {:?}", a);
    println!("Tensor b: {:?}", b);
    let u = new_from_array(&[2i8, 5i8, 90i8, -23i8]);
    println!("Tensor u: {:?}", u);
}
