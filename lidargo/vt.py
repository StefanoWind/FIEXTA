import matplotlib.pyplot as plt

def rwsfig(self):
    fig_rws = plt.figure(figsize=(18, 8))
    ctr = 1
    for i in i_plot:
        time = (ds.time[:, i] - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(
            1, "s"
        )

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

        xlim = [np.min(ds.x), np.max(ds.x)]
        ylim = [np.min(ds.y), np.max(ds.y)]
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
        xlim = [np.min(ds.x), np.max(ds.x)]
        ylim = [np.min(ds.y), np.max(ds.y)]
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


def rwsFig2(self):
    fig_rws = plt.figure(figsize=(18, 8))
    ctr = 1
    for i in i_plot:
        time = (ds.time[:, i] - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(
            1, "s"
        )

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

        xlim = [np.min(ds.x), np.max(ds.x)]
        zlim = [np.min(ds.z), np.max(ds.z)]
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


def rwsfig3(self):