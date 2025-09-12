from concurrent import futures

import numpy as np

from .utils import timer

# TODO: could optionall return an xarray dataset for the parameters


@timer
def get_from_segment_params(
    tosegment: np.ndarray,
    parallel=False,
    check=False,
) -> dict[str, np.ndarray]:
    """Get from segment parameters.

    This solves an unstructured/sparse array representation for the one to many
    upstream tracing by index. These parameters are required to use the
    get_nhm_segs_ids_above_seg function also in this module.

    For a given index in tosegment (related to its collated nhm_seg), the set
    of upstream indices from which this segment recives flow is given in Python
    by the code

    from_segment[from_segment_starts[ind]: from_segment_ends[ind] + 1]

    Note that the returned parameters are intended to match the assumptions of
    the original PRMS parameters in fortran: they are 1-based indexed and the
    from_segment_ends

    Args:
        tosegment: the tosegment parameter from the PRMS parameter files.
        parallel: Use concurrent futures ThreadPoolExecutor to solve in
            parallel. For large domains, this may save some time.
        check: Check the from solutions against the tosegment information.
    """
    # convert python indexing from fortran
    tosegment = tosegment.copy() - 1

    def get_from(index):
        wh_from = np.where(tosegment == index)[0]
        from_len = len(wh_from)
        return {index: (from_len, wh_from)}

    # <
    if parallel:
        with futures.ThreadPoolExecutor() as executor:
            results = executor.map(get_from, range(len(tosegment)))
            results = list(results)
    else:
        results = []
        for ii in range(len(tosegment)):
            results.append(get_from(ii))

    # <<
    inds = np.array([list(ii.keys())[0] for ii in results])
    lens = np.array([tuple(ii.values())[0][0] for ii in results])
    starts = np.concat([np.array([0]), np.cumsum(lens)[:-1]])
    ends = np.cumsum(lens)

    froms = [tuple(ii.values())[0][1].tolist() for ii in results]
    froms = np.array([jj for sublist in froms for jj in sublist])

    if check:
        assert (np.diff(inds) == 1).all()
        for ii in range(len(tosegment)):
            # finding "from" on fromcheck will give ii
            fromcheck = tosegment[ii]
            if fromcheck == -1:
                assert ii not in froms
            elif starts[fromcheck] == ends[fromcheck]:
                assert fromcheck == -1
            else:
                # msg = (
                #     f"{ii=} "
                #     f"{fromcheck=} "
                #     f"{starts[fromcheck]=} "
                #     f"{ends[fromcheck]=} "
                #     f"{froms[starts[fromcheck] : ends[fromcheck]]=} "
                # )
                msg = "oh no. A better error message is available in comments."
                assert ii in froms[starts[fromcheck] : ends[fromcheck]], msg
        print("From parameter checks passed.")

    # convert back to fortran indexing for consistency
    inds += 1
    starts += 1
    # ends += 1 # this would also subtract one for non-python slicing
    froms += 1

    return {
        "from_segment_starts": starts,
        "from_segment_ends": ends,
        "from_segment": froms,
    }


def get_nhm_segs_ids_above_seg(
    start_seg: np.int64,
    nhm_segs: np.ndarray,
    nhm_ids: np.ndarray,
    hru_segment_nhm: np.ndarray,
    from_segment_starts: np.ndarray,
    from_segment_ends: np.ndarray,
    from_segment: np.ndarray,
) -> dict[str, np.ndarray]:
    """Get nhm segments and hru ids above a given segment index.

    Args:
        start_seg: The starting segment index.
        nhm_segs: The array of segment ids.
        nhm_ids: The array of NHM IDs.
        hru_segment_nhm: Array mapping HRU to segments.
        from_segment_starts: Array of starting indices for each from_segment
            index.
        from_segment_ends: Array of ending indices for each from_segment index.
        from_segment: The unstructured array saying where flow comes from for
            a particular segment.
    """
    start_seg_ind = np.where(nhm_segs == start_seg)[0].tolist()[0]
    # pyton indexing
    starts = from_segment_starts - 1
    # no -1 on ends for python slicing
    ends = from_segment_ends
    froms = from_segment - 1

    all_inds = []
    current_inds = [start_seg_ind]
    while len(current_inds):
        new_inds = []
        for cc in current_inds:
            new_inds += froms[starts[cc] : ends[cc]].tolist()
        # <
        all_inds += current_inds
        current_inds = new_inds

    nhm_segs_above = nhm_segs[np.array(all_inds)]
    # solve the hrus from the identified nhm_segs
    nhm_ids_above = nhm_ids[np.where(np.isin(hru_segment_nhm, nhm_segs_above))]

    return {
        "nhm_ids_above": nhm_ids_above,
        "nhm_segs_above": nhm_segs_above,
    }
