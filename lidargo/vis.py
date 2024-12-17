import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
from lidargo import utilities
import os
import warnings
from lidargo.utilities import with_logging

# Suppress all UserWarnings containing 'interpreted as cell centers' for pcolormesh and pcolor
warnings.filterwarnings(
    "ignore",
    message=".*interpreted as cell centers, but are not monotonically increasing or decreasing.*",
)


def probabilityScatter(ds, ax=None, cax=None, fig=None, **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (9, 8)))

    prob_arange = np.arange(
        np.log10(ds.qc_wind_speed.attrs["min_probability_range"]) - 1,
        np.log10(ds.qc_wind_speed.attrs["max_probability_range"]) + 1,
    )

    sp = ax.scatter(
        ds.rws_norm,
        ds.snr_norm,
        s=3,
        c=np.log10(ds.probability),
        cmap=kwargs.get("cmap", "hot"),
        vmin=prob_arange[0],
        vmax=prob_arange[-1],
    )
    ax.set_xlabel("Normalized radial wind speed [m s$^{-1}$]")
    ax.set_ylabel("Normalized SNR [dB]")
    ax.grid(True)

    if cax is None:
        cax = fig.add_axes(
            [
                ax.get_position().x0 + ax.get_position().width + 0.01,
                ax.get_position().y0,
                0.015,
                ax.get_position().height,
            ]
        )
    cbar = plt.colorbar(sp, cax=cax, label="Probability")
    cbar.set_ticks(prob_arange)
    cbar.set_ticklabels([r"$10^{" + str(int(p)) + "}$" for p in prob_arange])
    # fig.tight_layout()
    return ax


# Second function: Semilog plot
def probabilityVSrws(ds, qc_rws_range, ax=None, fig=None, **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (9, 8)))

    ax.semilogx(
        ds.probability.values.ravel(),
        ds.rws_norm.values.ravel(),
        ".k",
        alpha=kwargs.get("alpha", 0.01),
        label=kwargs.get("label_all", "All points"),
    )
    ax.semilogx(
        [10**b.mid for b in qc_rws_range.index],
        qc_rws_range.values,
        ".b",
        markersize=kwargs.get("markersize", 10),
        label=kwargs.get("label_range", "Range"),
    )
    ax.semilogx(
        [
            ds["qc_wind_speed"].attrs["qc_probability_threshold"],
            ds["qc_wind_speed"].attrs["qc_probability_threshold"],
        ],
        [
            -ds["qc_wind_speed"].attrs["rws_norm_limit"],
            ds["qc_wind_speed"].attrs["rws_norm_limit"],
        ],
        "--r",
        label=kwargs.get("label_threshold", "Threshold"),
    )

    ax.set_xlabel("Probability")
    ax.set_ylabel("Normalized radial wind speed [m s$^{-1}$]")
    ax.yaxis.set_label_position("right")
    ax.legend()
    ax.grid(True)

    return ax


def rws(ds, ax=None, fig=None, cax=None, **kwargs):
    """
    Plot radial wind speed (RWS) data in different projections based on scan type.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing the RWS data.
    """

    if ds.attrs["scan_mode"].lower() == "ppi":
        fig, ax, cbar = ppi(ds, ax=ax, fig=fig, cax=cax)
    elif ds.attrs["scan_mode"].lower() == "rhi":
        fig, ax, cbar = rhi(ds, ax=ax, fig=fig, cax=cax)
    elif ds.attrs["scan_mode"].lower() == "volumetric":
        fig, ax, cbar = plot_volumetric(ds, ax=ax, fig=fig, cax=cax)
    elif ds.attrs["scan_mode"].lower() == "stare":
        fig, ax, cbar = stare(ds, ax=ax, fig=fig, cax=cax)
    else:
        raise ValueError(f"Unsupported scan type: {ds.attrs['scan_mode']}")

    return fig, ax, cbar


def ppi(ds, n_subplots: int = 5, ax=None, fig=None, cax=None, **kwargs):
    """Plot plan position indicator (PPI) for radial wind speed data."""

    if fig is None or ax is None:
        fig, ax = plt.subplots(
            1,
            n_subplots,
            sharex=True,
            sharey=True,
            figsize=kwargs.get("figsize", (9, 8)),
        )

    scans = np.linspace(
        ds.scanID.values.min(), ds.scanID.values.max(), n_subplots, dtype=int
    )

    for i, scan in enumerate(scans):
        subset = ds.isel(scanID=scan)
        im = ax[i].pcolormesh(
            subset.x, subset.y, subset.wind_speed, cmap="RdBu_r", shading="auto"
        )
        ax[i].set_xlabel(r"$x$ [m]")
        if i == 0:
            ax[i].set_ylabel(r"$y$ [m]")
        ax[i].set_aspect("equal")
        add_time_title(ax[i], ds.time.sel(scanID=scan))

    if cax is None:
        fig.tight_layout()
        cbar = add_colorbar(fig, ax[-1], im, "Radial Wind \n"+r"Speed [m s${-1}$]")
    else:
        cbar = plt.colorbar(im, cax=cax, label="Radial Wind \n"+r"Speed [m s${-1}$]")

    return fig, ax, cbar


def stare(ds):
    """Plot stare visualization for radial wind speed data."""
    # Implement stare plotting logic here
    raise NotImplementedError("Stare plotting is not implemented yet.")


@with_logging
def azimuthScatter(dsInput, dsStandardized, ax=None, fig=None, **kwargs):
    """scatter plot showing observed and standardized azimuth angles"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (9, 8)))

    ax.scatter(
        dsInput.time,
        dsInput.azimuth,
        marker=".",
        c="C0",
        label="input data",
    )
    ax.scatter(
        dsStandardized.time,
        dsStandardized.azimuth,
        marker=".",
        c="C1",
        label="regularized data",
    )

    xformatter = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(xformatter)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Azimuth Angle [˚]")

    ax.set_title(
        titleGenerator(
            dsStandardized,
            "Beam angles",
            ["location", "date", "file"],
        )
    )

    return fig, ax


def azimuthhist(ds, ax=None, fig=None, rwidth=0.8, **kwargs):
    """bar plot of occurrences of azimuth angles"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5, 3)))

    ax.hist(ds.azimuth.values.flatten(), bins=len(ds.beamID), rwidth=rwidth, **kwargs)
    ax.set_xlabel("Azimuth Angle [˚]")
    ax.set_ylabel("Occurrence")

    return fig, ax


def azimuthDeltaHist(ds1, ds2, bins: int = 10, ax=None, fig=None, **kwargs):
    """plot a histogram of recorded vs standardized azimuth angles"""

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5, 3)))

    az = {}
    for ds in [ds1, ds2]:
        if len(ds.azimuth.coords) != 1:
            az["standardized"] = ds.azimuth.stack(time=["scanID", "beamID"]).values
        else:
            az["input"] = ds.azimuth.values

    deltaAzimuth = az["input"] - az["standardized"]

    ax.hist(deltaAzimuth, bins=bins, **kwargs)
    plt.xticks(rotation=30)
    ax.set_xlabel("Change in Azimuth Angle [˚]")
    ax.set_ylabel("Occurrence")

    return fig, ax


def rws3Dscatter(ax, x, y, z, f, real, skip):
    """Helper function for 3D scatter plotting."""
    sc = ax.scatter(
        x[real][::skip],
        y[real][::skip],
        z[real][::skip],
        s=2,
        c=f[real][::skip],
        cmap="coolwarm",
        vmin=np.nanpercentile(f, 5) - 1,
        vmax=np.nanpercentile(f, 95) + 1,
    )

    xlim = [np.min(x), np.max(x)]
    ylim = [np.min(y), np.max(y)]
    zlim = [np.min(z), np.max(z)]

    if ctr == 1:
        ax.set_xlabel(r"$x$ [m]", labelpad=10)
        ax.set_ylabel(r"$y$ [m]")
        ax.set_zlabel(r"$z$ [m]")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ax.set_box_aspect((np.diff(xlim)[0], np.diff(ylim)[0], np.diff(zlim)[0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    return sc


def add_colorbar(fig, ax, mappable, label, position_adjust=0.01):
    """Helper function to add colorbar."""
    cax = fig.add_axes(
        [
            ax.get_position().x0 + ax.get_position().width + position_adjust,
            ax.get_position().y0,
            0.015,
            ax.get_position().height,
        ]
    )
    return plt.colorbar(mappable, cax=cax, label=label)


def titleGenerator(ds, inputString: str = None, components=["location", "date"]):
    """build title for figures"""
    componentDict = {
        "date": f"on {ds.time.dt.strftime('%Y-%m-%d').values.flatten()[0]}",
        "location": f"at {ds.attrs['location_id']}",
        "file": f"\nFile: {ds.attrs['datastream']}",
    }

    title = [inputString] + [componentDict[comp] for comp in components]

    return " ".join(title)


def add_main_title(fig, time, location_id, filename):
    """Helper function to add main title."""
    fig.suptitle(
        f"Radial wind speed at {location_id} on "
        f"{utilities.datestr(np.nanmean(time), '%Y-%m-%d')}\n"
        f"File: {filename}\n"
        f"{utilities.datestr(np.nanmin(time), '%H:%M:%S')} - "
        f"{utilities.datestr(np.nanmax(time), '%H:%M:%S')}"
    )


def add_time_title(ax, time):
    """Helper function to add time-only title."""
    ax.set_title(
        f"{time.min().dt.strftime('%H:%M:%S').values} - "
        f"{time.max().dt.strftime('%H:%M:%S').values}"
    )


def rhi(ds, n_subplots: int = 5, ax=None, fig=None, cax=None, **kwargs):
    """Plot range height indicator (RHI) for radial wind speed data."""
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            1,
            n_subplots,
            sharex=True,
            sharey=True,
            figsize=kwargs.get("figsize", (9, 8)),
        )

    scans = np.linspace(
        ds.scanID.values.min(), ds.scanID.values.max(), n_subplots, dtype=int
    )

    for i, scan in enumerate(scans):
        subset = ds.isel(scanID=scan)
        im = ax[i].pcolormesh(
            subset.x, subset.z, subset.wind_speed, cmap="RdBu_r", shading="auto"
        )
        ax[i].set_xlabel(r"$x$ [m]")
        if i == 0:
            ax[i].set_ylabel(r"$z$ [m]")
        ax[i].set_aspect("equal")
        add_time_title(ax[i], ds.time.sel(scanID=scan))

    if cax is None:
        fig.tight_layout()
        cbar = add_colorbar(fig, ax[-1], im, "Radial Wind \n"+r"Speed [m s${-1}$]")
    else:
        cbar = plt.colorbar(im, cax=cax, label="Radial Wind \n"+r"Speed [m s${-1}$]")

    return fig, ax, cbar

    # raise NotImplementedError("RHI plotting is not implemented yet.")


def plot_volumetric(ds):
    """Plot volumetric visualization for radial wind speed data."""
    # Implement volumetric plotting logic here
    raise NotImplementedError("Volumetric plotting is not implemented yet.")


def volumetric(ds):
    """Plot 3D visualization of radial wind speed data."""
    fig = plt.figure(figsize=(18, 9))
    ctr = 1

    for i in range(min(5, len(ds.scanID))):
        time = (ds.time[:, i] - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(
            1, "s"
        )
        x, y, z = [ds[coord][:, :, i].values for coord in ["x", "y", "z"]]

        for subplot_idx, (data, label) in enumerate(
            [(ds.rws, "Raw"), (ds.rws_qc, "Filtered")]
        ):
            f = data[:, :, i].values
            real = ~np.isnan(x + y + z + f)

            # Subsample if too many points
            skip = int(np.sum(real) / 10000) if np.sum(real) > 10000 else 1

            ax = plt.subplot(
                2,
                min(5, len(ds.scanID)),
                ctr + subplot_idx * min(5, len(ds.scanID)),
                projection="3d",
            )
            sc = rws3Dscatter(ax, x, y, z, f, real, skip)

            if ctr == np.ceil(min(5, len(ds.scanID)) / 2) and subplot_idx == 0:
                add_main_title(
                    fig,
                    time,
                    ds.attrs["location_id"],
                    os.path.basename(ds.attrs["datastream"]),
                )
            elif subplot_idx == 0:
                add_time_title(ax, time)

            if ctr == min(5, len(ds.scanID)):
                add_colorbar(
                    fig,
                    ax,
                    sc,
                    f"{label} radial\n wind speed [m s$^{-1}$]",
                    position_adjust=0.035,
                )

        ctr += 1

    plt.subplots_adjust(
        left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.25
    )
    return fig


@with_logging
def windSpeedQCfig(ds, qc_rws_range):
    """wrapper method to make qc and probability scatter plots"""
    fig = plt.figure(figsize=(10, 4), layout="constrained")
    gs = GridSpec(nrows=1, ncols=3, width_ratios=[6, 0.25, 6], figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[0, 2])

    probabilityScatter(ds, ax=ax0, fig=fig, cax=cax)
    probabilityVSrws(ds, qc_rws_range, ax=ax1, fig=fig)

    fig.suptitle(
        titleGenerator(ds, "QC of data", components=["location", "date", "file"])
    )

    return fig


@with_logging
def scanFig(ds):
    """wrapper method to make scans pcolor figures"""
    fig = plt.figure(figsize=(12, 3), layout="constrained")
    gs = GridSpec(nrows=2, ncols=6, width_ratios=[6, 6, 6, 6, 6, 0.5], figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    axt = [ax1] + [
        fig.add_subplot(gs[0, x], sharex=ax1, sharey=ax1) for x in range(1, 5)
    ]
    axb = [ax2] + [
        fig.add_subplot(gs[1, x], sharex=ax2, sharey=ax2) for x in range(1, 5)
    ]
    caxt = fig.add_subplot(gs[0, -1])
    caxb = fig.add_subplot(gs[1, -1])

    fig, axt, _ = rws(ds, fig=fig, ax=axt, cax=caxt)

    b1qc = ds.copy()
    b1qc["wind_speed"] = b1qc["wind_speed"].where(
        (b1qc.qc_wind_speed == 0.0)
    )  # & (b1qc.range <= 1000.0)
    fig, axb, _ = rws(b1qc, fig=fig, ax=axb, cax=caxb)

    return fig


@with_logging
def azHistFig(ds, dsInput):
    """wrapper method to make azimuth histograms"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    fig, _ = azimuthhist(ds, ax=ax[0], fig=fig, rwidth=0.7, color=".25")
    fig, _ = azimuthDeltaHist(dsInput, ds, ax=ax[1], fig=fig, color="C2")

    return fig


def qcReport(ds, dsInput, qc_rws_range):
    """wrapper method to make qc figures"""
    wsqc_fig = windSpeedQCfig(ds, qc_rws_range)
    scanqc_fig = scanFig(ds)
    az_fig, _ = azimuthScatter(dsInput, ds, figsize=(10, 3))
    azhist_fig = azHistFig(ds, dsInput)

    return wsqc_fig, scanqc_fig, az_fig, azhist_fig
