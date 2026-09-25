def smooth_web_risk(
    risk,
    valid_mask,
    radius,
):

    if radius <= 0:

        output = np.array(
            risk,
            dtype="float32",
            copy=True,
        )

        output[
            ~valid_mask
        ] = np.nan

        return output

    # --------------------------------------------------------
    # CRITICAL FIX
    #
    # valid_mask alone is not enough.
    # Some cells can be inside Fars but still contain NaN
    # because the source FWI/risk has no valid value there.
    #
    # NaN values must NOT enter the cumulative sum.
    # --------------------------------------------------------

    usable = (
        valid_mask
        &
        np.isfinite(
            risk
        )
    )

    values = np.where(
        usable,
        risk,
        0.0,
    ).astype(
        "float32"
    )

    weights = usable.astype(
        "float32"
    )

    # --------------------------------------------------------
    # Weighted smoothing
    # --------------------------------------------------------

    numerator = box_blur_2d(
        values,
        radius,
    )

    denominator = box_blur_2d(
        weights,
        radius,
    )

    output = np.full(
        risk.shape,
        np.nan,
        dtype="float32",
    )

    good = (
        denominator > 0.001
    )

    output[
        good
    ] = (
        numerator[
            good
        ]
        /
        denominator[
            good
        ]
    )

    output = np.clip(
        output,
        0.0,
        100.0,
    )

    # --------------------------------------------------------
    # FINAL FARS MASK
    #
    # Nothing outside the Fars boundary can become visible.
    # --------------------------------------------------------

    output[
        ~valid_mask
    ] = np.nan

    return output
