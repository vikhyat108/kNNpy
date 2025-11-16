import os
import sys
import pytest
import numpy as np
from scipy.interpolate import interp1d

#The module that needs to be tested

#Necessary for relative imports (see https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im)
module_path = os.path.abspath(os.path.join('../'))           # '../' is needed because the parent directory is one directory upstream of this test directory
if module_path not in sys.path:
    sys.path.append(module_path)

from kNNpy.HelperFunctions import calc_kNN_CDF, cdf_vol_knn, CIC_3D_Interp, create_query_3D, create_smoothed_field_dict_3D, smoothing_3D

#--------------------------------------------------------------------------------------------------------------------------

#calc_kNN_CDF()

def test_basic_calc_kNN_CDF():
    """Basic 2×2 example: check analytical CDF values at given bins."""
    vol = np.array([[1.0, 2.0],
                    [3.0, 4.0]])
    bins = [
        np.array([0., 1., 2., 3., 4.]),
        np.array([0., 2., 3., 4., 5.])
    ]
    result = calc_kNN_CDF(vol, bins)
    # Expect two arrays
    assert isinstance(result, list) and len(result) == 2
    # k=1: vol[:,0]=[1,3] → CDF=[0,0.5,0.75,1,1]
    np.testing.assert_allclose(result[0], [0., 0.5, 0.75, 1., 1.])
    # k=2: vol[:,1]=[2,4] → CDF=[0,0.5,0.75,1,1]
    np.testing.assert_allclose(result[1], [0., 0.5, 0.75, 1., 1.])

def test_bins_below_minimum():
    """All bins below the minimum distance should yield zeros."""
    vol = np.array([[1.], [2.], [3.]])
    bins = [np.array([-10., -5., 0.])]
    result = calc_kNN_CDF(vol, bins)
    assert len(result) == 1
    assert np.all(result[0] == 0.0)

def test_bins_above_maximum():
    """All bins above the maximum distance should yield ones."""
    vol = np.array([[1.], [2.], [3.]])
    bins = [np.array([10., 20., 30.])]
    result = calc_kNN_CDF(vol, bins)
    assert len(result) == 1
    assert np.all(result[0] == 1.0)

def test_bin_at_exact_maximum():
    """Bins exactly equal to max_dist should be included in interpolation, not clipped."""
    vol = np.array([[1.], [2.], [3.]])
    bins = [np.array([1., 3., 5.])]
    # CDF at 1→1/3, at 3→1, at 5→1
    expected = np.array([1/3, 1.0, 1.0])
    result = calc_kNN_CDF(vol, bins)
    np.testing.assert_allclose(result[0], expected, rtol=1e-6)

def test_interpolation_midpoints():
    """Check linear interpolation produces correct midpoint CDF."""
    vol = np.array([[2.], [4.]])
    bins = [np.array([2., 3., 4.])]
    # gof = [0.5, 1.0] → at r=3: 0.5 + (3-2)*(1-0.5)/(4-2) = 0.75
    expected = np.array([0.5, 0.75, 1.0])
    result = calc_kNN_CDF(vol, bins)
    np.testing.assert_allclose(result[0], expected, rtol=1e-6)

def test_invalid_bins_length_raises_index_error():
    """If bins list is shorter than number of kNN columns, should raise IndexError."""
    vol = np.array([[1.0, 2.0], [3.0, 4.0]])
    bins = [np.array([0., 1., 2., 3., 4.])]  # missing second column
    with pytest.raises(IndexError):
        calc_kNN_CDF(vol, bins)

def test_extra_bins_ignored():
    """If bins list is longer than kNN columns, extra entries are ignored."""
    vol = np.array([[1.0, 2.0], [3.0, 4.0]])
    bins = [
        np.array([0., 1., 2., 3., 4.]),
        np.array([0., 2., 3., 4., 5.]),
        np.array([10., 20., 30.])  # extra—should be ignored
    ]
    result = calc_kNN_CDF(vol, bins)
    assert len(result) == 2  # only two columns of `vol` produce output

def test_random_vol_returns_correct_shape_and_range():
    """Random vol + uniform bins should return CDF values in [0,1] of correct shape."""
    rng = np.random.default_rng(42)
    vol = rng.uniform(0, 10, size=(5, 3))
    vol = np.sort(vol, axis=0)
    bins = [np.linspace(0, 10, 6) for _ in range(3)]
    result = calc_kNN_CDF(vol, bins)
    assert isinstance(result, list) and len(result) == 3
    for arr, b in zip(result, bins):
        assert arr.shape == (len(b),)
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0)

def test_negative_distances_and_bins():
    """Negative distances with matching negative bins handled correctly."""
    vol = np.array([[-3.], [0.], [3.]])
    bins = [np.array([-5., -3., 0., 3., 5.])]
    # CDF: at -3→1/3, 0→2/3, 3→1; below -3→0, above 3→1
    expected = np.array([0., 1/3, 2/3, 1.0, 1.0])
    result = calc_kNN_CDF(vol, bins)
    np.testing.assert_allclose(result[0], expected, rtol=1e-6)

def test_mismatched_vol_bins_empty_vol():
    """An empty vol (0×n_kNN) should return an empty list (no columns to iterate)."""
    result = calc_kNN_CDF(np.empty((0, 3)), [np.array([0., 1.])] * 3)
    assert result == []

# Two possible behaviors for empty-vol input:
# 1. Return an empty list: []
# 2. Raise a ValueError
# Which behavior should our code implement?

#--------------------------------------------------------------------------------------------------------------------------

#cdf_vol_kNN()

def test_basic_two_by_two():
    """Exact CDF values at sample points for a simple 2×2 input."""
    vol = np.array([[1., 2.],
                    [3., 4.]])
    cdfs = cdf_vol_knn(vol)
    assert isinstance(cdfs, list) and len(cdfs) == 2
    assert all(isinstance(fn, interp1d) for fn in cdfs)

    # k=1: [1→0.5, 3→1.0]; k=2: [2→0.5, 4→1.0]
    np.testing.assert_allclose(cdfs[0]([1., 3.]), [0.5, 1.0])
    np.testing.assert_allclose(cdfs[1]([2., 4.]), [0.5, 1.0])

def test_unsorted_input_sorted_internally():
    """Columns get sorted before interpolation, even if input is unsorted."""
    vol = np.array([[3., 4.],
                    [1., 2.]])
    cdfs = cdf_vol_knn(vol)
    np.testing.assert_allclose(cdfs[0]([1., 3.]), [0.5, 1.0])
    np.testing.assert_allclose(cdfs[1]([2., 4.]), [0.5, 1.0])

def test_monotonicity_on_support():
    """Each CDF must be non-decreasing on its own [min, max] domain."""
    vol = np.array([[1., 10.],
                    [2., 20.],
                    [3., 30.]])
    cdfs = cdf_vol_knn(vol)
    for i, fn in enumerate(cdfs):
        # build x within [min_i, max_i]
        col = vol[:, i]
        mn, mx = col.min(), col.max()
        x = np.linspace(mn, mx, 6)
        y = fn(x)
        # no NaNs on support, and non-decreasing
        assert not np.any(np.isnan(y))
        assert np.all(np.diff(y) >= -1e-8)

def test_bounds_return_nan():
    """Out-of-bounds queries yield NaN (bounds_error=False)."""
    vol = np.array([[5.],[15.]])
    fn = cdf_vol_knn(vol)[0]
    assert np.isnan(fn(4.))   # below min
    assert np.isnan(fn(16.))  # above max
    # but at exactly min/max, you still get a value
    assert fn(5.) == pytest.approx(1/2)
    assert fn(15.) == pytest.approx(1.0)

def test_empty_input_returns_empty_list():
    """An empty (0×0) input should simply return [] (no error)."""
    result = cdf_vol_knn(np.empty((0,0)))
    assert result == []

def test_single_query_case():
    """With n=1, CDF equals 1.0 at the sample and NaN elsewhere."""
    vol = np.array([[2., 3., 5.]])
    cdfs = cdf_vol_knn(vol)
    for i, fn in enumerate(cdfs):
        r = vol[0, i]
        assert fn(r) == pytest.approx(1.0)
        assert np.isnan(fn(r + 1.0))   # anything > r is NaN

def test_invalid_nd_input_raises():
    """A 1D array should fail (no second dimension)."""
    with pytest.raises(Exception):
        cdf_vol_knn(np.array([1.,2.,3.]))

def test_mixed_type_arrays_work():
    """Integer inputs are cast to float and still produce correct CDFs."""
    vol = np.array([[1,2],
                    [3,4]], dtype=int)
    cdfs = cdf_vol_knn(vol)
    np.testing.assert_allclose(cdfs[0]([1,3]), [0.5,1.0])
    np.testing.assert_allclose(cdfs[1]([2,4]), [0.5,1.0])

def test_negative_distances():
    """Negative distances sort correctly and yield a valid CDF."""
    vol = np.array([[-5.],[5.]])
    fn = cdf_vol_knn(vol)[0]
    assert fn(-5.) == pytest.approx(0.5)
    assert fn(5.)  == pytest.approx(1.0)

def test_large_k_many_columns():
    """For a (10×50) array, we get back 50 interpolators."""
    vol = np.random.rand(10,50)
    cdfs = cdf_vol_knn(vol)
    assert isinstance(cdfs, list) and len(cdfs)==50

#--------------------------------------------------------------------------------------------------------------------------

#CIC_3D_Interp()

def test_constant_field_interpolation():
    """A constant field should interpolate to the same constant value at any positions."""
    Ng = 5
    Boxsize = 10
    field = np.ones((Ng, Ng, Ng), dtype=np.float32)
    pos = np.random.rand(20, 3) * Boxsize
    pos = pos.astype(np.float32)
    out = CIC_3D_Interp(pos, field, Boxsize)
    # Shape & dtype
    assert out.shape == (20,)
    assert out.dtype == np.float32
    # All values should equal 1.0
    assert np.allclose(out, 1.0)

def test_interpolation_at_grid_nodes():
    """Interpolation at exact grid nodes should return the field value at those nodes."""
    Ng = 4
    Boxsize = 4
    cell_size = Boxsize / Ng
    # Make a field whose value encodes its (i,j,k) index
    field = np.arange(Ng**3).reshape(Ng, Ng, Ng).astype(np.float32)
    # Pick a few grid‐node indices
    idxs = [(0,0,0), (1,2,3), (3,3,3)]
    # Convert to physical positions
    pos = np.array([(i*cell_size, j*cell_size, k*cell_size) for i,j,k in idxs], dtype=np.float32)
    out = CIC_3D_Interp(pos, field, Boxsize)
    expected = np.array([field[i,j,k] for i,j,k in idxs], dtype=np.float32)
    np.testing.assert_allclose(out, expected)

def test_empty_positions_returns_empty():
    """Passing an empty positions array should return an empty interpolation array."""
    field = np.random.rand(3,3,3).astype(np.float32)
    pos = np.empty((0,3)).astype(np.float32)
    out = CIC_3D_Interp(pos, field, 3.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)
    assert out.dtype == np.float32

def test_invalid_positions_dimension_raises():
    """Passing pos with shape (N,2) should raise an error due to dimension mismatch."""
    field = np.ones((3,3,3), dtype=np.float32)
    pos = np.zeros((5,2), dtype=float)
    with pytest.raises(Exception):
        CIC_3D_Interp(pos, field, 3.0)

def test_non_cubical_field_raises():
    """A non-cubical field should cause an error in the underlying CIC routine."""
    field = np.zeros((4,4,3), dtype=np.float32)
    pos = np.zeros((1,3))
    with pytest.raises(Exception):
        CIC_3D_Interp(pos, field, 4.0)

#--------------------------------------------------------------------------------------------------------------------------

#create_query_3D()

def test_grid_shape_and_unique_values():
    """Grid mode: shape should be (n³, 3) and coords drawn from linspace(0, BoxSize, n)."""
    n, Box = 4, 8.0
    pts = create_query_3D('grid', n, Box)
    # Check shape
    assert pts.shape == (n**3, 3)
    # All unique coordinates along any axis should be exactly the linspace values
    xs = np.unique(pts[:,0])
    expected = np.linspace(0.0, Box, n, endpoint=False)
    np.testing.assert_allclose(xs, expected)

# def test_grid_corners_for_n2(): # this test is no longer valid because of endpoint=False
#     """For n=2, grid points must be exactly the 8 corners of the cube [0,Box]^3."""
#     n, Box = 2, 10.0
#     pts = create_query_3D('grid', n, Box)
#     expected = {
#         (0.0,0.0,0.0), (0.0,0.0,Box),
#         (0.0,Box,0.0), (0.0,Box,Box),
#         (Box,0.0,0.0), (Box,0.0,Box),
#         (Box,Box,0.0), (Box,Box,Box),
#     }
#     assert set(map(tuple, pts.round(6))) == expected

def test_single_grid_point():
    """n=1 should produce a single point at the origin."""
    pts = create_query_3D('grid', 1, 42.0)
    assert pts.shape == (1,3)
    np.testing.assert_allclose(pts, [[0.0,0.0,0.0]])

def test_random_shape_and_bounds():
    """Random mode: shape (n³,3), and all coordinates in [0, BoxSize)."""
    n, Box = 3, 5.0
    pts = create_query_3D('random', n, Box)
    assert pts.shape == (n**3, 3)
    assert np.all(pts >= 0.0) and np.all(pts < Box)

def test_random_reproducibility_with_seed():
    """Resetting np.random.seed should reproduce the same random points."""
    n, Box = 3, 7.0
    np.random.seed(1234)
    pts1 = create_query_3D('random', n, Box)
    np.random.seed(1234)
    pts2 = create_query_3D('random', n, Box)
    np.testing.assert_allclose(pts1, pts2)

def test_invalid_query_type_raises():
    """Passing any other query_type must raise ValueError."""
    with pytest.raises(ValueError) as exc:
        create_query_3D('foo', 3, 10.0)
    assert 'Unknown query type' in str(exc.value)

def test_zero_query_grid_grid_mode():
    """n=0 in grid mode should yield an empty (0,3) array, not an error."""
    pts = create_query_3D('grid', 0, 5.0)
    assert isinstance(pts, np.ndarray)
    assert pts.shape == (0, 3)

def test_zero_query_grid_random_mode():
    """n=0 in random mode should yield an empty (0,3) array, not an error."""
    pts = create_query_3D('random', 0, 5.0)
    assert isinstance(pts, np.ndarray)
    assert pts.shape == (0, 3)

# def test_negative_boxsize_grid(): #This test is no longer necessary because we have included a ValueError for BoxSize<0 in create_query_3D()
#     """Negative BoxSize in grid mode should still produce linspace from 0 to BoxSize."""
#     n, Box = 3, -6.0
#     pts = create_query_3D('grid', n, Box)
#     # unique x coordinates should be linspace(0, -6, 3)
#     expected = np.linspace(0.0, Box, n)
#     np.testing.assert_allclose(np.unique(pts[:,0]), expected)

# def test_negative_boxsize_grid(): #See comment above
#     """Negative BoxSize in grid mode should still produce linspace from 0 to BoxSize."""
#     n, Box = 3, -6.0
#     pts = create_query_3D('grid', n, Box)
#     # The unique x-coordinates come back sorted ascending: [-6., -3., 0.]
#     actual = np.unique(pts[:, 0])
#     # The expected linspace is [0., -3., -6.], so sort it to match actual
#     expected = np.sort(np.linspace(0.0, Box, n))
#     np.testing.assert_allclose(actual, expected)

#--------------------------------------------------------------------------------------------------------------------------

#create_smoothed_field_dict_3D()

def test_invalid_filter_top_hat_k():
    """Using 'Top-Hat-k' should immediately raise ValueError."""
    field = np.zeros((4,4,4))
    bins = [np.array([1.0, 2.0])]
    with pytest.raises(ValueError) as exc:
        create_smoothed_field_dict_3D(field, 'Top-Hat-k', 4, 10.0, bins)
    assert "Unknown filter" in str(exc.value)

def test_empty_bins_list_returns_empty_dict():
    """Empty bins list yields an empty dictionary (no smoothing done)."""
    field = np.zeros((4,4,4))
    result = create_smoothed_field_dict_3D(field, 'Gaussian', 4, 10.0, [])
    assert isinstance(result, dict)
    assert result == {}

def test_single_bin_key_and_shape():
    """Single-bin list creates one key 'R' with array of correct shape & dtype."""
    N = 5
    field = np.random.rand(N,N,N).astype(np.float32)
    R = 3.5
    bins = [np.array([R])]
    result = create_smoothed_field_dict_3D(field, 'Gaussian', N, 20.0, bins)
    # Expect one entry with key '3.5'
    assert list(result.keys()) == [str(R)]
    out = result[str(R)]
    assert isinstance(out, np.ndarray)
    assert out.shape == (N,N,N)

def test_multiple_bins_multiple_keys():
    """Multiple sublists produce all radii as string keys."""
    N = 3
    field = np.zeros((N,N,N))
    bins = [
        np.array([1.0, 2.0]),
        np.array([5.0])
    ]
    result = create_smoothed_field_dict_3D(field, 'Gaussian', N, 30.0, bins)
    expected_keys = { '1.0', '2.0', '5.0' }
    assert set(result.keys()) == expected_keys
    # All outputs should have the same shape as field
    for key, arr in result.items():
        assert arr.shape == field.shape

def test_uniform_field_gaussian_returns_uniform():
    """Uniform field smoothed with Gaussian remains (approximately) uniform."""
    N = 6
    field = np.ones((N,N,N), dtype=np.float32)
    bins = [np.array([2.0, 4.0])]
    result = create_smoothed_field_dict_3D(field, 'Gaussian', N, 50.0, bins)
    for key, arr in result.items():
        # Gaussian smoothing of a constant field should return the same constant
        assert np.allclose(arr, np.ones_like(field), rtol=1e-6)

def test_missing_thickness_propagates_error():
    """Omitting thickness for 'Shell' should raise ValueError from smoothing_3D."""
    N = 4
    field = np.zeros((N,N,N))
    bins = [np.array([1.0, 2.0])]
    with pytest.raises(ValueError) as exc:
        create_smoothed_field_dict_3D(field, 'Shell', N, 10.0, bins)
    # The underlying smoothing_3D should complain about missing thickness
    assert "thickness" in str(exc.value)

def test_verbose_prints(capsys):
    """Verbose=True should print summary messages."""
    N = 4
    field = np.zeros((N,N,N))
    bins = [np.array([1.0])]
    _ = create_smoothed_field_dict_3D(field, 'Gaussian', N, 10.0, bins, Verbose=True)
    captured = capsys.readouterr().out
    assert "Smoothing the density field" in captured
    assert "Total time taken" in captured


# In our test `test_single_bin_key_and_shape`, we asserted that the smoothed field should
# have the exact same shape as the input (grid×grid×grid). The error occurred because
# calling `irfftn(field_k)` without specifying the output shape makes FFTW infer the
# last dimension from the real-to-complex transform, which for odd grid sizes (e.g. 5) gives
# an output of shape (5,5,4) instead of (5,5,5). The unit test caught this mismatch,
# revealing that we need to explicitly pass `s=(grid, grid, grid)` to `irfftn` so the
# smoothed field always comes back at the correct shape.

#--------------------------------------------------------------------------------------------------------------------------

#smoothing_3D()

# ——— Error‐handling tests —————————————————————————————————————————————————————————

def test_non_cubical_error():
    """Non‐cubic input array should raise ValueError."""
    field = np.zeros((4, 4, 3))
    with pytest.raises(ValueError) as exc:
        smoothing_3D(field, 'Gaussian', 4, 100.0, R=1.0)
    assert "not cubical" in str(exc.value)

def test_grid_mismatch_error():
    """Grid parameter not matching field.shape should raise ValueError."""
    field = np.zeros((5, 5, 5))
    with pytest.raises(ValueError) as exc:
        smoothing_3D(field, 'Top-Hat', 4, 100.0, R=1.0)
    assert "Grid size provided" in str(exc.value)

def test_missing_R_real_space_error():
    """Missing R for a real-space filter ('Top-Hat') should raise ValueError."""
    field = np.zeros((4, 4, 4))
    with pytest.raises(ValueError) as exc:
        smoothing_3D(field, 'Top-Hat', 4, 100.0)
    assert "R must be provided" in str(exc.value)

def test_missing_kmin_kmax_error():
    """Missing kmin/kmax for 'Top-Hat-k' should raise ValueError."""
    field = np.zeros((4, 4, 4))
    with pytest.raises(ValueError) as exc:
        smoothing_3D(field, 'Top-Hat-k', 4, 100.0)
    assert "Both kmin and kmax must be provided" in str(exc.value)

def test_missing_thickness_error():
    """Missing thickness for 'Shell' should raise ValueError."""
    field = np.zeros((4, 4, 4))
    with pytest.raises(ValueError) as exc:
        smoothing_3D(field, 'Shell', 4, 100.0, R=2.0)
    assert "Both R and thickness must be provided" in str(exc.value)

def test_unknown_filter_error():
    """Unknown filter string should raise ValueError."""
    field = np.zeros((4, 4, 4))
    with pytest.raises(ValueError) as exc:
        smoothing_3D(field, 'Foo', 4, 100.0, R=1.0)
    assert "Unknown filter" in str(exc.value)


# ——— Warning tests —————————————————————————————————————————————————————————

def test_real_space_unused_args_warn():
    """Passing kmin/kmax/thickness into a real-space filter should warn."""
    field = np.zeros((4, 4, 4))
    with pytest.warns(UserWarning) as record:
        smoothing_3D(field, 'Gaussian', 4, 100.0,
                     R=1.0, kmin=0.1, kmax=1.0, thickness=0.5)
    msgs = [str(w.message) for w in record]
    assert any("kmin and kmax are not used" in m for m in msgs)
    assert any("thickness is not used" in m for m in msgs)

def test_k_space_unused_args_warn():
    """Passing R/thickness into 'Top-Hat-k' should warn."""
    field = np.zeros((4, 4, 4))
    with pytest.warns(UserWarning) as record:
        smoothing_3D(field, 'Top-Hat-k', 4, 100.0,
                     kmin=0.1, kmax=1.0, R=2.0, thickness=0.5)
    msgs = [str(w.message) for w in record]
    assert any("R is not used" in m for m in msgs)
    assert any("thickness is not used" in m for m in msgs)

def test_shell_unused_kspace_args_warn():
    """Passing kmin/kmax into 'Shell' should warn."""
    field = np.zeros((4, 4, 4))
    with pytest.warns(UserWarning) as record:
        smoothing_3D(field, 'Shell', 4, 100.0,
                     R=2.0, thickness=0.5, kmin=0.1, kmax=1.0)
    msgs = [str(w.message) for w in record]
    assert any("kmin and kmax are not used" in m for m in msgs)


# ——— Shape & behavior tests ———————————————————————————————————————————————————————

@pytest.mark.parametrize("filt,params", [
    ('Top-Hat',    dict(R=5.0)),
    ('Gaussian',   dict(R=5.0)),
    ('Top-Hat-k',  dict(kmin=0.1, kmax=1.0)),
    ('Shell',      dict(R=3.0, thickness=1.0)),
])
def test_valid_filters_return_shape_and_dtype(filt, params):
    """Each valid filter returns an array of the same shape & dtype as input."""
    field = np.random.rand(4, 4, 4).astype(np.float32)
    out = smoothing_3D(field, filt, 4, 100.0, **params)
    assert isinstance(out, np.ndarray)
    assert out.shape == field.shape
    assert out.dtype in (np.float32, np.float64)

def test_uniform_field_shell_may_be_all_nan():
    """On a small grid the Shell window can have zero support, yielding all NaNs."""
    N = 4
    field = np.ones((N, N, N), dtype=np.float32)
    with pytest.warns(RuntimeWarning):
        sm = smoothing_3D(field, 'Shell', N, 50.0, R=10.0, thickness=2.0)
    assert np.isnan(sm).all()

def test_verbose_prints(capsys):
    """Verbose=True should print timing messages to stdout."""
    field = np.zeros((4, 4, 4))
    _ = smoothing_3D(field, 'Gaussian', 4, 100.0, R=1.0, Verbose=True)
    captured = capsys.readouterr().out
    assert "Starting smoothing" in captured
    assert "Smoothing completed." in captured