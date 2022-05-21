use std::any::Any;
use tensorism::dimensions::*;
use tensorism::*;

#[test]
fn dimension_types_are_distinct() {
    let n = 5;
    let d1 = new_dynamic_dim!(n);
    let d2 = new_dynamic_dim!(n);
    assert!(d1.type_id() != d2.type_id());
}

#[test]
fn dimensions_are_ok() {
    let d1 = new_dynamic_dim!(5);
    let d2 = new_dynamic_dim!(5);
    let static_d = new_static_dim::<7>();

    assert_eq!(5, d1.as_usize());
    assert_eq!(5, d2.as_usize());
    assert_eq!(7, static_d.as_usize());

    assert_eq!("5|f091", format!("{:?}", d1));
    assert_eq!("5|8853", format!("{:?}", d2));
    assert_eq!("7", format!("{:?}", static_d));

    let static_d = new_static_dim::<5>();
    assert_eq!(static_d, d1);
}
