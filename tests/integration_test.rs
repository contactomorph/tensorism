use std::any::Any;
use tensorism::building::{Shape, TensorBuilder, TensorBuilding};
use tensorism::dimensions::*;
use tensorism::tensors::{StaticMatrix, Tensor};
use tensorism::*;

#[test]
fn dimension_types_are_distinct() {
    let n = 5;
    let d1 = new_dynamic_dim!(n);
    let d2 = new_dynamic_dim!(n);
    assert_ne!(d1.type_id(), d2.type_id());
}

#[test]
fn dimensions_are_ok() {
    let d1 = new_dynamic_dim!(5);
    let d2 = new_dynamic_dim!(5);
    let static_d = new_static_dim::<7>();

    assert_eq!(5, d1.as_usize());
    assert_eq!(5, d2.as_usize());
    assert_eq!(7, static_d.as_usize());

    assert_ne!(format!("{:?}", d1), format!("{:?}", d2));
    assert_eq!("7", format!("{:?}", static_d));

    let static_d = new_static_dim::<5>();
    assert_eq!(static_d, d1);
}

#[test]
fn tensor_equality() {
    let s = "Hello world".to_owned();
    let a = TensorBuilding::with_static::<8>()
        .with_static::<8>()
        .fill(&s);
    let b: StaticMatrix<8, 8, String> = a.clone();
    assert_eq!(a, b);

    let d = new_dynamic_dim!(4);
    let c = TensorBuilding::with(d).with_first().fill(&7);

    assert_eq!(
        format!(
            "〈4•{0:04x}, 4•{0:04x}〉[7, 7, 7, 7 | 7, 7, 7, 7 | 7, 7, …]",
            d.get_thumbprint().unwrap()
        ),
        format!("{:?}", c)
    );
}

#[test]
fn building_shape() {
    let d = new_dynamic_dim!(3);
    let s = TensorBuilding::with(d).with_static::<5>().with_first();

    assert_eq!(45, s.count());
    assert_eq!(
        format!("〈3•{0:04x}, 5, 3•{0:04x}〉", d.get_thumbprint().unwrap()),
        format!("{:?}", s)
    );
}

#[test]
fn comparing_shapes() {
    let s1 = TensorBuilding::with(new_dynamic_dim!(4)).with_static::<4>();
    let s2 = s1.switch_12();
    assert_eq!(s1, s2);
    assert_ne!(format!("{:?}", s1), format!("{:?}", s2));
}

#[test]
fn generate_tensor() {
    let a = TensorBuilding::with_static::<3>()
        .with_static::<5>()
        .define(|(i, j)| (3 * i + 2 * j) % 7);

    assert_eq!(
        "〈3, 5〉[0, 2, 4, 6, 1 | 3, 5, 0, 2, 4 | 6, 1, 3, 5, 0]",
        format!("{:#?}", a)
    );
    assert_eq!(6, a[(0, 3)]);
    assert_eq!(4, a[(1, 4)]);
    assert_eq!(1, a[(2, 1)]);

    let mut a = a;
    a.update(|(i, j), element| {
        if i == j {
            *element = 110
        }
    });
    assert_eq!(110, a[(0, 0)]);
    assert_eq!(110, a[(1, 1)]);
    assert_eq!(110, a[(2, 2)]);

    let shape = TensorBuilding::with_static::<3>()
        .with_static::<2>()
        .with_static::<2>();

    let b = shape.define(|(i, j, k)| (3 * i + 2 * j + 5 * k) % 17);

    assert_eq!(
        "〈3, 2, 2〉[0, 5 | 2, 7 || 3, 8 | 5, 10 || 6, 11 | 8, 13]",
        format!("{:#?}", b)
    );

    let c = shape
        .prepare()
        .append_vec(&mut vec![0, 5, 2, 7, 3, 8, 5, 10, 6, 11, 8, 13])
        .generate();

    assert_eq!(b, c);
    assert_eq!(7, c[(0, 1, 1)]);
    assert_eq!(5, c[(1, 1, 0)]);
    assert_eq!(11, c[(2, 0, 1)]);
}
