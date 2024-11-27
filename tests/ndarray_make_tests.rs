use std::str::FromStr;
use ndarray::{Array1, Array2};
use tensorism::{ndarray_format_for_make, ndarray_make};

#[test]
fn format_make_macro_with_single_tensor() {
    let string = ndarray_format_for_make! {(i j $ a[i, j] + i as f64).sum()};
    assert_eq!(
        "{ \
            let i_dimension : usize = :: ndarray :: ArrayBase :: < _, _ > :: dim(& a).0; \
            let j_dimension : usize = :: ndarray :: ArrayBase :: < _, _ > :: dim(& a).1; \
            ((0usize .. i_dimension).flat_map(move | i | (0usize .. j_dimension).map(move | j | (i, j,)))\
            .map(| (i, j,) | { (* unsafe { a.uget((i, j,)) }) + i as f64 })).sum() \
        } ",
        string
    );
}

#[test]
fn format_make_macro_with_dimensions_consistency() {
    let string = ndarray_format_for_make! {i $ (j $ a[i, j] + b[j])};
    assert_eq!(
        "{ \
            let i_dimension : usize = :: ndarray :: ArrayBase :: < _, _ > :: dim(& a).0; \
            let j_dimension : usize = :: ndarray :: ArrayBase :: < _, _ > :: dim(& a).1; \
            { \
                let first : usize = :: ndarray :: ArrayBase :: < _, _ > :: dim(& a).1; \
                let second : usize = :: ndarray :: ArrayBase :: < _, _ > :: dim(& b); \
                if first != second { panic! (\"Non matching dimensions\") } \
            } \
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 1usize] >> :: from_shape_fn((i_dimension,), | (i,) | { \
                ((0usize .. j_dimension).map(move | j | (j,))\
                .map(| (j,) | { (* unsafe { a.uget((i, j,)) }) + (* unsafe { b.uget(j) }) })) \
            }) \
        } ",
        string);
}

fn count_all_chars<'a>(it: impl Iterator<Item = &'a String>) -> usize {
    it.fold(0usize, |acc, message| acc + message.chars().count())
}

#[test]
fn run_make_macro_aggregating_on_single_index() {
    let messages = ["Hello", "World", "How", "are you?"].map(|s| String::from_str(s).unwrap());
    let c: Array1<_> = Array1::<String>::from_iter(messages);
    let all_chars_count = ndarray_make! {count_all_chars(i $ &c[i])};
    assert_eq!(21, all_chars_count);
}

#[test]
fn run_make_macro_aggregating_twice_on_each_indexes() {
    let a: Array2<_> = Array2::<_>::from_shape_fn((9, 10), |(i, j)| {
        i as i64 * (j + 1) as i64
    });
    let x: i64 = ndarray_make! {Iterator::sum(i $ Iterator::min(j $ a[i, j]).unwrap())};
    assert_eq!(36i64, x);
}

#[test]
fn run_make_macro_creating_new_tensor() {
    let a: Array2<_> = Array2::<_>::from_shape_fn((9, 10), |(i, j)| {
        i as i64 * (j + 1) as i64
    });
    let b: Array1<_> = Array1::<f64>::from_elem((10,),12f64);
    let t = ndarray_make! {i j $ a[i, j] as f64 + b[j]};
    assert_eq!((9, 10), t.dim());
}

#[test]
#[should_panic(expected = "Non matching dimensions")]
fn run_make_macro_must_fail() {
    let a: Array2<_> = Array2::<_>::from_shape_fn((9, 10), |(i, j)| {
        i as i64 * (j + 1) as i64
    });
    let b: Array1<_> = Array1::<f64>::from_elem((50,),12f64);
    ndarray_make! {i j $ a[i, j] as f64 + b[j]};
}
