import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_gantt_multiple(
        work_records,
        row_labels=None,
        start_time=0,
        end_time=None,
        color_mode="auto",
        color="C0",
        save_path=None
    ):
    num_rows = len(work_records)
    if row_labels is None:
        row_labels = [f"Device {i}" for i in range(num_rows)]
    if end_time is None:
        end_time = max([max([r[1] for r in records]) for records in work_records])

    split_points = set([start_time, end_time])
    for records in work_records:
        for s, e, _ in records:
            if start_time <= s <= end_time:
                split_points.add(s)
            if start_time <= e <= end_time:
                split_points.add(e)
    split_points = sorted(split_points)

    all_tags = set()
    for records in work_records:
        for r in records:
            all_tags.add(r[2])
    all_tags = sorted(all_tags)
    if color_mode == "single":
        tag_colors = {tag: color for tag in all_tags}
    else:
        tag_colors = {tag: f"C{i%10}" for i, tag in enumerate(all_tags)}

    split_point_gap = 3
    min_width = 18
    max_width = 50
    line_height = 0.65
    min_height = 5
    save_dpi = 300
    height = max(min_height, num_rows * line_height + 2.5)

    max_splits_per_page = int(max_width / split_point_gap)
    if max_splits_per_page < 2:
        max_splits_per_page = 2
    n_pages = max(1, (len(split_points) + max_splits_per_page - 1) // max_splits_per_page)

    for page_idx in range(n_pages):
        sp_start = page_idx * max_splits_per_page
        sp_end = min(sp_start + max_splits_per_page, len(split_points))
        page_splits = split_points[sp_start:sp_end]
        page_t0 = page_splits[0]
        page_t1 = page_splits[-1]

        width = min(max(min_width, split_point_gap * len(page_splits)), max_width)
        fig, ax = plt.subplots(figsize=(width, height))

        bar_height = 0.65
        label_fontsize = 11
        split_fontsize = 11

        for i, records in enumerate(work_records):
            for s, e, tag in records:
                if e <= page_t0 or s >= page_t1:
                    continue
                bs = max(s, page_t0)
                be = min(e, page_t1)
                ax.broken_barh(
                    [(bs, be - bs)], (i - bar_height / 2, bar_height),
                    facecolors=tag_colors[tag], edgecolors='black', alpha=0.92
                )
                lx = (bs + be) / 2
                ly = i
                if be - bs < 0.07 * (page_t1 - page_t0):
                    ly = i + 0.32
                ax.text(lx, ly, tag, va='center', ha='center', color='black',
                        fontsize=label_fontsize, fontweight='bold', zorder=5,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.74, pad=0.06))

        for x in page_splits:
            ax.axvline(x, color='gray', linestyle='--', linewidth=1, alpha=0.63, zorder=0)
            ax.text(x, -1.13, f'{x:g}', va='top', ha='center', fontsize=split_fontsize, color='gray',
                    rotation=40 if width < 30 else 0, zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.14'))

        ax.set_yticks(range(num_rows))
        ax.set_yticklabels(row_labels, fontsize=label_fontsize)
        ax.set_xlabel("")
        ax.set_title(f"Page {page_idx+1}/{n_pages}" if n_pages > 1 else "")
        ax.set_xlim(page_t0, page_t1)
        ax.set_ylim(-1.3, num_rows - 0.5 + bar_height / 2)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_xticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        if save_path:
            if n_pages > 1:
                base, ext = os.path.splitext(save_path)
                page_path = f"{base}_page{page_idx}{ext}"
            else:
                page_path = save_path
            plt.savefig(page_path, dpi=save_dpi)
        else:
            plt.show()
        plt.close(fig)


def tuple_to_structured_str(t):
    if isinstance(t, tuple):
        return '(' + ','.join(tuple_to_structured_str(i) for i in t) + ')'
    else:
        return str(t)


def devicedict_to_showlist(
    device_dict,
    records_lists=None,
    device_names=None
    ):

    if records_lists is None:
        records_lists = []
        device_names = []

    for device_tag in device_dict:
        device_names.append(tuple_to_structured_str(device_tag))
        str_work_record = [[l[0], l[1], tuple_to_structured_str(l[2])] for l in device_dict[device_tag].work_record]
        records_lists.append(str_work_record)

    return records_lists, device_names


if __name__ == "__main__":

    test_tag = ((1, 0), (1, 1), 2)
    print(tuple_to_structured_str(test_tag))


    work_records = [
        [[0, 5, "load"], [6, 8, "compute"], [8, 12, "comm"], [15, 20, "compute"]],
        [[0, 3, "comm"], [3, 9, "compute"], [10, 12, "load"], [14, 16, "comm"]],
        [[1, 4, "load"], [4, 7, "compute"], [11, 18, "comm"]]
    ]
    row_labels = ["Worker A", "Worker B", "Worker C"]
    start_time = 5
    end_time = 15

    plot_gantt_multiple(
        work_records,
        row_labels,
        start_time,
        end_time,
        show_legend=False,
    )
