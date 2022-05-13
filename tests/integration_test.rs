use lazy_static::lazy_static;
use tensorism::*;
use tensorism::dimensions::*;

#[test]
fn dimensions_are_ok() {
    let n = 3 + "AA".len();
    let d = new_dim!(n);
    let static_d = new_static_dim::<7>();
    assert_eq!(5, d.as_usize());
    assert_eq!(7, static_d.as_usize());
    let fd = format!("{:?}", d);
    assert!(fd.starts_with("5|"));
    assert_eq!("7", format!("{:?}", static_d));

    let static_d = new_static_dim::<5>();
    assert_eq!(static_d, d);
}
