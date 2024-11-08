import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from lidargo import utilities


class LidarVisualizer:
    def plot_rws(self, plot_type="xy"):
        """
        Plot 2D radial wind speed (RWS) data in different projections.

        Parameters:
        -----------
        plot_type : str
            Type of plot projection to use: 'xy' for horizontal, 'xz' for vertical
        """
        fig_rws = plt.figure(figsize=(18, 8))
        ctr = 1

        for i in i_plot:
            time = (
                ds.time[:, i] - np.datetime64("1970-01-01T00:00:00")
            ) / np.timedelta64(1, "s")

            # Set axes based on plot type
            if plot_type == "xy":
                X2, Y2 = X, Y
                y_label = r"$y$ [m]"
                ylim = [np.min(ds.y), np.max(ds.y)]
            else:  # xz projection
                X2, Y2 = X, Z
                y_label = r"$z$ [m]"
                ylim = [np.min(ds.z), np.max(ds.z)]

            xlim = [np.min(ds.x), np.max(ds.x)]

            # Plot raw and QC'ed RWS
            for subplot_idx, (data, label) in enumerate(
                [(rws, "Raw"), (rws_qc, "Filtered")]
            ):
                ax = plt.subplot(2, N_plot, ctr + subplot_idx * N_plot)
                pc = plt.pcolor(
                    X2,
                    Y2,
                    data[:, :, i],
                    cmap="coolwarm",
                    vmin=np.nanpercentile(rws_qc, 5) - 1,
                    vmax=np.nanpercentile(rws_qc, 95) + 1,
                )

                if ctr == 1:
                    plt.ylabel(y_label)
                else:
                    ax.set_yticklabels([])

                if subplot_idx == 1:
                    plt.xlabel(r"$x$ [m]")

                plt.grid()
                ax.set_box_aspect(np.diff(ylim) / np.diff(xlim))
                plt.xlim(xlim)
                plt.ylim(ylim)

                # Add title and colorbar
                if ctr == np.ceil(N_plot / 2) and subplot_idx == 0:
                    self._add_main_title(time)
                elif subplot_idx == 0:
                    self._add_time_title(time)

                if ctr == N_plot:
                    self._add_colorbar(
                        fig_rws, ax, pc, f"{label} radial\n wind speed [m s$^{-1}$]"
                    )

            ctr += 1

        return fig_rws

    def plot_beam_angles(self):
        """Plot beam angles analysis including azimuth, elevation, and occurrence."""
        fig_angles = plt.figure(figsize=(18, 10))

        # Azimuth time series
        plt.subplot(4, 1, 1)
        self._plot_angle_timeseries("azimuth")
        plt.ylabel(r"Azimuth [$^\circ$]")
        self._add_main_title(self.inputData["time"])

        # Elevation time series
        plt.subplot(4, 1, 2)
        self._plot_angle_timeseries("elevation")
        plt.xlabel("Time (UTC)")
        plt.ylabel(r"Elevation [$^\circ$]")

        # Azimuth vs Elevation occurrence plot
        ax = plt.subplot(2, 1, 2)
        plt.plot(self.azimuth_detected, self.elevation_detected, "--k")
        sc = plt.scatter(
            self.azimuth_detected, self.elevation_detected, c=self.counts, cmap="hot"
        )
        plt.xlabel(r"Azimuth [$^\circ$]")
        plt.ylabel(r"Elevation [$^\circ$]")
        plt.grid()
        ax.set_facecolor("lightgrey")
        self._add_colorbar(fig_angles, ax, sc, "Occurrence")

        return fig_angles

    def plot_3d_rws(self):
        """Plot 3D visualization of radial wind speed data."""
        fig_rws = plt.figure(figsize=(18, 9))
        ctr = 1

        for i in i_plot:
            time = (
                self.outputData.time[:, i] - np.datetime64("1970-01-01T00:00:00")
            ) / np.timedelta64(1, "s")
            x, y, z = [
                self.outputData[coord][:, :, i].values for coord in ["x", "y", "z"]
            ]

            # Plot raw and QC'ed RWS in 3D
            for subplot_idx, (data, label) in enumerate(
                [(rws, "Raw"), (rws_qc, "Filtered")]
            ):
                f = data[:, :, i].values
                real = ~np.isnan(x + y + z + f)

                # Subsample if too many points
                skip = int(np.sum(real) / 10000) if np.sum(real) > 10000 else 1

                ax = plt.subplot(2, N_plot, ctr + subplot_idx * N_plot, projection="3d")
                sc = self._plot_3d_scatter(ax, x, y, z, f, real, skip)

                if ctr == np.ceil(N_plot / 2) and subplot_idx == 0:
                    self._add_main_title(time)
                elif subplot_idx == 0:
                    self._add_time_title(time)

                if ctr == N_plot:
                    self._add_colorbar(
                        fig_rws,
                        ax,
                        sc,
                        f"{label} radial\n wind speed [m s$^{-1}$]",
                        position_adjust=0.035,
                    )

            ctr += 1

        plt.subplots_adjust(
            left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.25
        )
        return fig_rws

    def _plot_angle_timeseries(self, angle_type):
        """Helper function to plot angle time series."""
        plt.plot(
            self.inputData["time"],
            self.inputData[angle_type].values.ravel(),
            ".k",
            markersize=5,
            label=f"Raw {angle_type}",
        )
        plt.plot(
            getattr(self, f"{angle_type}_selected").time,
            getattr(self, f"{angle_type}_selected"),
            ".r",
            markersize=5,
            label=f"Selected {angle_type}",
        )
        plt.plot(
            getattr(self, f"{angle_type}_regularized").time,
            getattr(self, f"{angle_type}_regularized"),
            ".g",
            markersize=5,
            label=f"Regularized {angle_type}",
        )
        date_fmt = mdates.DateFormatter("%H:%M:%S")
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.legend()
        plt.grid()

    def _plot_3d_scatter(self, ax, x, y, z, f, real, skip):
        """Helper function for 3D scatter plotting."""
        sc = ax.scatter(
            x[real][::skip],
            y[real][::skip],
            z[real][::skip],
            s=2,
            c=f[real][::skip],
            cmap="coolwarm",
            vmin=np.nanpercentile(rws_qc, 5) - 1,
            vmax=np.nanpercentile(rws_qc, 95) + 1,
        )

        xlim = [np.min(self.outputData.x), np.max(self.outputData.x)]
        ylim = [np.min(self.outputData.y), np.max(self.outputData.y)]
        zlim = [np.min(self.outputData.z), np.max(self.outputData.z)]

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

    def _add_colorbar(self, fig, ax, mappable, label, position_adjust=0.01):
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

    def _add_main_title(self, time):
        """Helper function to add main title."""
        plt.title(
            f"Radial wind speed at {self.inputData.attrs['location_id']} on "
            f"{utilities.datestr(np.nanmean(time), '%Y-%m-%d')}\n"
            f"File: {os.path.basename(self.source)}\n"
            f"{utilities.datestr(np.nanmin(time), '%H:%M:%S')} - "
            f"{utilities.datestr(np.nanmax(time), '%H:%M:%S')}"
        )

    def _add_time_title(self, time):
        """Helper function to add time-only title."""
        plt.title(
            f"{utilities.datestr(np.nanmin(time), '%H:%M:%S')} - "
            f"{utilities.datestr(np.nanmax(time), '%H:%M:%S')}"
        )
