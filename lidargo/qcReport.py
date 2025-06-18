import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

from lidargo import utilities


class QCReport:
    def __init__(self, ds) -> None:

        pass

    def qc_report(self):
        """
        Make figures.
        """

        rws = self.outputData["wind_speed"].copy()
        rws_qc = self.outputData["wind_speed"].where(self.outputData.qc_wind_speed == 0)
        X = self.outputData["x"].mean(dim="scanID")
        Y = self.outputData["y"].mean(dim="scanID")
        Z = self.outputData["z"].mean(dim="scanID")
        r = self.outputData["range"]

        fig_probability = plt.figure(figsize=(18, 8))
        prob_arange = np.arange(
            np.log10(self.min_probability_range) - 1,
            np.log10(self.max_probability_range) + 1,
        )
        ax1 = plt.subplot(1, 2, 1)
        sp = plt.scatter(
            self.outputData.rws_norm,
            self.outputData.snr_norm,
            s=3,
            c=np.log10(self.outputData.probability),
            cmap="hot",
            vmin=prob_arange[0],
            vmax=prob_arange[-1],
        )
        plt.xlabel("Normalized radial wind speed [m s$^{-1}$]")
        plt.ylabel("Normalized SNR [dB]")
        plt.grid()
        plt.title(
            "QC of data at "
            + self.inputData.attrs["location_id"]
            + " on "
            + utilities.datestr(
                np.nanmean(utilities.dt64_to_num(self.inputData["time"])), "%Y-%m-%d"
            )
            + "\n File: "
            + os.path.basename(self.source)
        )

        ax2 = plt.subplot(1, 2, 2)
        plt.semilogx(
            self.outputData.probability.values.ravel(),
            self.outputData.rws_norm.values.ravel(),
            ".k",
            alpha=0.01,
            label="All points",
        )
        plt.semilogx(
            [10**b.mid for b in self.qc_rws_range.index],
            self.qc_rws_range,
            ".b",
            markersize=10,
            label="Range",
        )
        plt.semilogx(
            [self.qc_probability_threshold, self.qc_probability_threshold],
            [-self.rws_norm_limit, self.rws_norm_limit],
            "--r",
            label="Threshold",
        )
        plt.xlabel("Probability")
        plt.ylabel("Normalized radial wind speed [m s$^{-1}$]")
        ax2.yaxis.set_label_position("right")
        plt.legend()
        plt.grid()

        fig_probability.subplots_adjust(left=0.1, right=0.9, wspace=0.3)
        cax = fig_probability.add_axes(
            [
                ax1.get_position().x0 + ax1.get_position().width + 0.01,
                ax1.get_position().y0,
                0.015,
                ax1.get_position().height,
            ]
        )
        cbar = plt.colorbar(sp, cax=cax, label="Probability")
        cbar.set_ticks(prob_arange)
        cbar.set_ticklabels([r"$10^{" + str(int(p)) + "}$" for p in prob_arange])

        N_plot = np.min(np.array([5, len(self.outputData.scanID)]))
        i_plot = [
            int(i) for i in np.linspace(0, len(self.outputData.scanID) - 1, N_plot)
        ]

        if self.outputData.attrs["scan_mode"] == "Stare":
            time = (
                self.outputData.time - np.datetime64("1970-01-01T00:00:00")
            ) / np.timedelta64(1, "s")
            fig_rws = plt.figure(figsize=(18, 9))
            ax = plt.subplot(2, 1, 1)
            pc = plt.pcolor(
                self.outputData["time"],
                r,
                rws[:, 0, :],
                cmap="coolwarm",
                vmin=np.nanpercentile(rws_qc, 5) - 1,
                vmax=np.nanpercentile(rws_qc, 95) + 1,
            )
            plt.xlabel("Time (UTC)")
            plt.ylabel(r"Range [m]")
            date_fmt = mdates.DateFormatter("%H:%M:%S")
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.grid()
            plt.title(
                "Radial wind speed at "
                + self.inputData.attrs["location_id"]
                + " on "
                + utilities.datestr(np.nanmean(time), "%Y-%m-%d")
                + "\n File: "
                + os.path.basename(self.source)
                + "\n"
                + utilities.datestr(np.nanmin(time), "%H:%M:%S")
                + " - "
                + utilities.datestr(np.nanmax(time), "%H:%M:%S")
            )

            cax = fig_rws.add_axes(
                [
                    ax.get_position().x0 + ax.get_position().width + 0.01,
                    ax.get_position().y0,
                    0.015,
                    ax.get_position().height,
                ]
            )
            cbar = plt.colorbar(
                pc, cax=cax, label="Raw radial \n" + r" wind speed [m s$^{-1}$]"
            )

            ax = plt.subplot(2, 1, 2)
            pc = plt.pcolor(
                self.outputData["time"],
                r,
                rws_qc[:, 0, :],
                cmap="coolwarm",
                vmin=np.nanpercentile(rws_qc, 5) - 1,
                vmax=np.nanpercentile(rws_qc, 95) + 1,
            )
            plt.xlabel("Time (UTC)")
            plt.ylabel(r"Range [m]")
            date_fmt = mdates.DateFormatter("%H:%M:%S")
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.grid()
            cax = fig_rws.add_axes(
                [
                    ax.get_position().x0 + ax.get_position().width + 0.01,
                    ax.get_position().y0,
                    0.015,
                    ax.get_position().height,
                ]
            )
            cbar = plt.colorbar(
                pc, cax=cax, label="Filtered radial \n" + r" wind speed [m s$^{-1}$]"
            )

        if self.outputData.attrs["scan_mode"] == "PPI":
            fig_angles = plt.figure(figsize=(18, 10))
            plt.subplot(2, 1, 1)
            plt.plot(
                self.inputData["time"],
                self.inputData["azimuth"].values.ravel(),
                ".k",
                markersize=5,
                label="Raw azimuth",
            )
            plt.plot(
                self.azimuth_selected.time,
                self.azimuth_selected,
                ".r",
                markersize=5,
                label="Selected azimuth",
            )
            plt.plot(
                self.azimuth_regularized.time,
                self.azimuth_regularized,
                ".g",
                markersize=5,
                label="Regularized azimuth",
            )
            plt.xlabel("Time (UTC)")
            plt.ylabel(r"Azimuth [$^\circ$]")
            date_fmt = mdates.DateFormatter("%H:%M:%S")
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.legend()
            plt.grid()
            plt.title(
                "Beam angles at "
                + self.inputData.attrs["location_id"]
                + " on "
                + utilities.datestr(
                    np.nanmean(utilities.dt64_to_num(self.inputData["time"])),
                    "%Y-%m-%d",
                )
                + "\n File: "
                + os.path.basename(self.source)
            )

            plt.subplot(2, 1, 2)
            plt.bar(self.azimuth_detected, self.counts, color="k")
            plt.xlabel(r"Detected significant azimuth [$^\circ$]")
            plt.ylabel("Occurrence")
            plt.grid()

            fig_rws = plt.figure(figsize=(18, 8))
            ctr = 1
            for i in i_plot:
                time = (
                    self.outputData.time[:, i] - np.datetime64("1970-01-01T00:00:00")
                ) / np.timedelta64(1, "s")

                # plot raw rws
                ax = plt.subplot(2, N_plot, ctr)
                pc = plt.pcolor(
                    X,
                    Y,
                    rws[:, :, i],
                    cmap="coolwarm",
                    vmin=np.nanpercentile(rws_qc, 5) - 1,
                    vmax=np.nanpercentile(rws_qc, 95) + 1,
                )
                if ctr == 1:
                    plt.ylabel(r"$y$ [m]")
                else:
                    ax.set_yticklabels([])
                plt.grid()

                xlim = [np.min(self.outputData.x), np.max(self.outputData.x)]
                ylim = [np.min(self.outputData.y), np.max(self.outputData.y)]
                ax.set_box_aspect(np.diff(ylim) / np.diff(xlim))
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.title(
                    utilities.datestr(np.nanmin(time), "%H:%M:%S")
                    + " - "
                    + utilities.datestr(np.nanmax(time), "%H:%M:%S")
                )

                if ctr == np.ceil(N_plot / 2):
                    plt.title(
                        "Radial wind speed at "
                        + self.inputData.attrs["location_id"]
                        + " on "
                        + utilities.datestr(np.nanmean(time), "%Y-%m-%d")
                        + "\n File: "
                        + os.path.basename(self.source)
                        + "\n"
                        + utilities.datestr(np.nanmin(time), "%H:%M:%S")
                        + " - "
                        + utilities.datestr(np.nanmax(time), "%H:%M:%S")
                    )

                if ctr == N_plot:
                    cax = fig_rws.add_axes(
                        [
                            ax.get_position().x0 + ax.get_position().width + 0.01,
                            ax.get_position().y0,
                            0.015,
                            ax.get_position().height,
                        ]
                    )
                    cbar = plt.colorbar(
                        pc, cax=cax, label="Raw radial \n" + r" wind speed [m s$^{-1}$]"
                    )

                # plot qc'ed rws
                ax = plt.subplot(2, N_plot, ctr + N_plot)
                plt.pcolor(
                    X,
                    Y,
                    rws_qc[:, :, i],
                    cmap="coolwarm",
                    vmin=np.nanpercentile(rws_qc, 5) - 1,
                    vmax=np.nanpercentile(rws_qc, 95) + 1,
                )
                plt.xlabel(r"$x$ [m]")
                if ctr == 1:
                    plt.ylabel(r"$y$ [m]")
                else:
                    ax.set_yticklabels([])
                xlim = [np.min(self.outputData.x), np.max(self.outputData.x)]
                ylim = [np.min(self.outputData.y), np.max(self.outputData.y)]
                ax.set_box_aspect(np.diff(ylim) / np.diff(xlim))
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.grid()

                if ctr == N_plot:
                    if ctr == N_plot:
                        cax = fig_rws.add_axes(
                            [
                                ax.get_position().x0 + ax.get_position().width + 0.01,
                                ax.get_position().y0,
                                0.015,
                                ax.get_position().height,
                            ]
                        )
                        cbar = plt.colorbar(
                            pc,
                            cax=cax,
                            label="Filtered radial \n" + r" wind speed [m s$^{-1}$]",
                        )
                ctr += 1

        if self.outputData.attrs["scan_mode"] == "RHI":
            fig_angles = plt.figure(figsize=(18, 10))
            plt.subplot(2, 1, 1)
            plt.plot(
                self.inputData["time"],
                self.inputData["elevation"].values.ravel(),
                ".k",
                markersize=5,
                label="Raw elevation",
            )
            plt.plot(
                self.elevation_selected.time,
                self.elevation_selected,
                ".r",
                markersize=5,
                label="Selected elevation",
            )
            plt.plot(
                self.elevation_regularized.time,
                self.elevation_regularized,
                ".g",
                markersize=5,
                label="Regularized elevation",
            )
            plt.xlabel("Time (UTC)")
            plt.ylabel(r"Elevation [$^\circ$]")
            date_fmt = mdates.DateFormatter("%H:%M:%S")
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.legend()
            plt.grid()
            plt.title(
                "Beam angles at "
                + self.inputData.attrs["location_id"]
                + " on "
                + utilities.datestr(
                    np.nanmean(utilities.dt64_to_num(self.inputData["time"])),
                    "%Y-%m-%d",
                )
                + "\n File: "
                + os.path.basename(self.source)
            )

            plt.subplot(2, 1, 2)
            plt.bar(self.elevation_detected, self.counts, color="k")
            plt.xlabel(r"Detected significant elevation [$^\circ$]")
            plt.ylabel("Occurrence")
            plt.grid()

            fig_rws = plt.figure(figsize=(18, 8))
            ctr = 1
            for i in i_plot:
                time = (
                    self.outputData.time[:, i] - np.datetime64("1970-01-01T00:00:00")
                ) / np.timedelta64(1, "s")

                # plot raw rws
                ax = plt.subplot(2, N_plot, ctr)
                pc = plt.pcolor(
                    X,
                    Z,
                    rws[:, :, i],
                    cmap="coolwarm",
                    vmin=np.nanpercentile(rws_qc, 5) - 1,
                    vmax=np.nanpercentile(rws_qc, 95) + 1,
                )
                if ctr == 1:
                    plt.ylabel(r"$z$ [m]")
                else:
                    ax.set_yticklabels([])
                plt.grid()

                xlim = [np.min(self.outputData.x), np.max(self.outputData.x)]
                zlim = [np.min(self.outputData.z), np.max(self.outputData.z)]
                ax.set_box_aspect(np.diff(zlim) / np.diff(xlim))
                plt.xlim(xlim)
                plt.ylim(zlim)
                plt.title(
                    utilities.datestr(np.nanmin(time), "%H:%M:%S")
                    + " - "
                    + utilities.datestr(np.nanmax(time), "%H:%M:%S")
                )

                if ctr == np.ceil(N_plot / 2):
                    plt.title(
                        "Radial wind speed at "
                        + self.inputData.attrs["location_id"]
                        + " on "
                        + utilities.datestr(np.nanmean(time), "%Y-%m-%d")
                        + "\n File: "
                        + os.path.basename(self.source)
                        + "\n"
                        + utilities.datestr(np.nanmin(time), "%H:%M:%S")
                        + " - "
                        + utilities.datestr(np.nanmax(time), "%H:%M:%S")
                    )

                if ctr == N_plot:
                    cax = fig_rws.add_axes(
                        [
                            ax.get_position().x0 + ax.get_position().width + 0.01,
                            ax.get_position().y0,
                            0.015,
                            ax.get_position().height,
                        ]
                    )
                    cbar = plt.colorbar(
                        pc, cax=cax, label="Raw radial \n" + r" wind speed [m s$^{-1}$]"
                    )

                # plot qc'ed rws
                ax = plt.subplot(2, N_plot, ctr + N_plot)
                plt.pcolor(
                    X,
                    Z,
                    rws_qc[:, :, i],
                    cmap="coolwarm",
                    vmin=np.nanpercentile(rws_qc, 5) - 1,
                    vmax=np.nanpercentile(rws_qc, 95) + 1,
                )
                plt.xlabel(r"$x$ [m]")
                if ctr == 1:
                    plt.ylabel(r"$z$ [m]")
                else:
                    ax.set_yticklabels([])

                ax.set_box_aspect(np.diff(zlim) / np.diff(xlim))
                plt.xlim(xlim)
                plt.ylim(zlim)
                plt.grid()

                if ctr == N_plot:
                    if ctr == N_plot:
                        cax = fig_rws.add_axes(
                            [
                                ax.get_position().x0 + ax.get_position().width + 0.01,
                                ax.get_position().y0,
                                0.015,
                                ax.get_position().height,
                            ]
                        )
                        cbar = plt.colorbar(
                            pc,
                            cax=cax,
                            label="Filtered radial \n" + r" wind speed [m s$^{-1}$]",
                        )
                ctr += 1

        if self.outputData.attrs["scan_mode"] == "3D":
            fig_angles = plt.figure(figsize=(18, 10))
            plt.subplot(4, 1, 1)
            plt.plot(
                self.inputData["time"],
                self.inputData["azimuth"].values.ravel(),
                ".k",
                markersize=5,
                label="Raw azimuth",
            )
            plt.plot(
                self.azimuth_selected.time,
                self.azimuth_selected,
                ".r",
                markersize=5,
                label="Selected azimuth",
            )
            plt.plot(
                self.azimuth_regularized.time,
                self.azimuth_regularized,
                ".g",
                markersize=5,
                label="Regularized azimuth",
            )
            plt.ylabel(r"Azimuth [$^\circ$]")
            date_fmt = mdates.DateFormatter("%H:%M:%S")
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.legend()
            plt.grid()
            plt.title(
                "Beam angles at "
                + self.inputData.attrs["location_id"]
                + " on "
                + utilities.datestr(
                    np.nanmean(utilities.dt64_to_num(self.inputData["time"])),
                    "%Y-%m-%d",
                )
                + "\n File: "
                + os.path.basename(self.source)
            )

            plt.subplot(4, 1, 2)
            plt.plot(
                self.inputData["time"],
                self.inputData["elevation"].values.ravel(),
                ".k",
                markersize=5,
                label="Raw elevation",
            )
            plt.plot(
                self.elevation_selected.time,
                self.elevation_selected,
                ".r",
                markersize=5,
                label="Selected elevation",
            )
            plt.plot(
                self.elevation_regularized.time,
                self.elevation_regularized,
                ".g",
                markersize=5,
                label="Regularized elevation",
            )
            plt.xlabel("Time (UTC)")
            plt.ylabel(r"Elevation [$^\circ$]")
            date_fmt = mdates.DateFormatter("%H:%M:%S")
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.legend()
            plt.grid()

            ax = plt.subplot(2, 1, 2)
            plt.plot(self.azimuth_detected, self.elevation_detected, "--k")
            sc = plt.scatter(
                self.azimuth_detected,
                self.elevation_detected,
                c=self.counts,
                cmap="hot",
            )
            plt.xlabel(r"Azimuth [$^\circ$]")
            plt.ylabel(r"Elevation [$^\circ$]")
            plt.grid()
            cax = fig_angles.add_axes(
                [
                    ax.get_position().x0 + ax.get_position().width + 0.01,
                    ax.get_position().y0,
                    0.015,
                    ax.get_position().height,
                ]
            )
            cbar = plt.colorbar(sc, cax=cax, label="Occurrence")
            ax.set_facecolor("lightgrey")

            fig_rws = plt.figure(figsize=(18, 9))
            ctr = 1
            for i in i_plot:
                time = (
                    self.outputData.time[:, i] - np.datetime64("1970-01-01T00:00:00")
                ) / np.timedelta64(1, "s")
                x = self.outputData.x[:, :, i].values
                y = self.outputData.y[:, :, i].values
                z = self.outputData.z[:, :, i].values
                f = rws[:, :, i].values
                real = ~np.isnan(x + y + z + f)

                if np.sum(real) > 10000:
                    skip = int(np.sum(real) / 10000)
                else:
                    skip = 1

                # plot raw rws
                xlim = [np.min(self.outputData.x), np.max(self.outputData.x)]
                ylim = [np.min(self.outputData.y), np.max(self.outputData.y)]
                zlim = [np.min(self.outputData.z), np.max(self.outputData.z)]

                ax = plt.subplot(2, N_plot, ctr, projection="3d")
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
                if ctr == 1:
                    ax.set_xlabel(r"$x$ [m]", labelpad=10)
                    ax.set_ylabel(r"$y$ [m]")
                    ax.set_zlabel(r"$z$ [m]")
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])

                ax.set_box_aspect(
                    (np.diff(xlim)[0], np.diff(ylim)[0], np.diff(zlim)[0])
                )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
                plt.title(
                    utilities.datestr(np.nanmin(time), "%H:%M:%S")
                    + " - "
                    + utilities.datestr(np.nanmax(time), "%H:%M:%S")
                )

                if ctr == np.ceil(N_plot / 2):
                    plt.title(
                        "Radial wind speed at "
                        + self.inputData.attrs["location_id"]
                        + " on "
                        + utilities.datestr(np.nanmean(time), "%Y-%m-%d")
                        + "\n File: "
                        + os.path.basename(self.source)
                        + "\n"
                        + utilities.datestr(np.nanmin(time), "%H:%M:%S")
                        + " - "
                        + utilities.datestr(np.nanmax(time), "%H:%M:%S")
                    )

                if ctr == N_plot:
                    cax = fig_rws.add_axes(
                        [
                            ax.get_position().x0 + ax.get_position().width + 0.035,
                            ax.get_position().y0,
                            0.015,
                            ax.get_position().height,
                        ]
                    )
                    cbar = plt.colorbar(
                        sc, cax=cax, label="Raw radial \n" + r" wind speed [m s$^{-1}$]"
                    )

                # plot qc'ed rws
                f = rws_qc[:, :, i].values
                real = ~np.isnan(x + y + z + f)
                ax = plt.subplot(2, N_plot, ctr + N_plot, projection="3d")
                sc = ax.scatter(
                    x[real],
                    y[real],
                    z[real],
                    s=2,
                    c=f[real],
                    cmap="coolwarm",
                    vmin=np.nanpercentile(rws_qc, 5) - 1,
                    vmax=np.nanpercentile(rws_qc, 95) + 1,
                )

                if ctr == 1:
                    ax.set_xlabel(r"$x$ [m]", labelpad=10)
                    ax.set_ylabel(r"$y$ [m]")
                    ax.set_zlabel(r"$z$ [m]")
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])

                ax.set_box_aspect(
                    (np.diff(xlim)[0], np.diff(ylim)[0], np.diff(zlim)[0])
                )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
                plt.grid()

                if ctr == N_plot:
                    if ctr == N_plot:
                        cax = fig_rws.add_axes(
                            [
                                ax.get_position().x0 + ax.get_position().width + 0.035,
                                ax.get_position().y0,
                                0.015,
                                ax.get_position().height,
                            ]
                        )
                        cbar = plt.colorbar(
                            sc,
                            cax=cax,
                            label="Filtered radial \n" + r" wind speed [m s$^{-1}$]",
                        )
                ctr += 1
            plt.subplots_adjust(
                left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.25
            )
        
        if not self.outputData.attrs["scan_mode"] == "Stare":
            fig_angles.savefig(self.save_filename.replace(self.save_filename.split('.')[-1], "_angles.png"))
        fig_probability.savefig(self.save_filename.replace(self.save_filename.split('.')[-1], "_probability.png"))
        fig_rws.savefig(self.save_filename.replace(self.save_filename.split('.')[-1], "_wind_speed.png"))

        self.print_and_log(f"QC figures saved in {os.path.dirname(self.save_filename)}")
