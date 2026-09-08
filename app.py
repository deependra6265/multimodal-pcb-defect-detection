import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from huggingface_hub import hf_hub_download

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔬",
    layout="wide"
)
MODEL_REPO = "deependra45/pcb-defect-unet"
MODEL_FILENAME = "improved_unet.keras"

PATCH_SIZE = 256
THRESHOLD = 0.45
MIN_AREA = 20
MERGE_DISTANCE = 20


# ============================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


model = load_model()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🔬 PCB Defect Detection System")

st.write(
    "Upload a PCB image to detect and analyze predicted defect regions."
)

st.info(
    "Model: U-Net | Patch Size: 256×256 | "
    f"Detection Threshold: {THRESHOLD}"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload PCB Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# FUNCTIONS
# ============================================================

def extract_patches(full_img):

    H, W, _ = full_img.shape

    x_positions = list(
        range(0, W - PATCH_SIZE + 1, PATCH_SIZE)
    )

    if x_positions[-1] != W - PATCH_SIZE:
        x_positions.append(W - PATCH_SIZE)

    y_positions = list(
        range(0, H - PATCH_SIZE + 1, PATCH_SIZE)
    )

    if y_positions[-1] != H - PATCH_SIZE:
        y_positions.append(H - PATCH_SIZE)

    patches = []
    positions = []

    for y in y_positions:

        for x in x_positions:

            patch = full_img[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            patches.append(patch)
            positions.append((x, y))

    return np.array(patches), positions


def create_probability_map(
    predictions,
    positions,
    H,
    W
):

    probability_map = np.zeros(
        (H, W),
        dtype=np.float32
    )

    count_map = np.zeros(
        (H, W),
        dtype=np.float32
    )

    for i, (x, y) in enumerate(positions):

        pred_patch = predictions[i].squeeze()

        probability_map[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ] += pred_patch

        count_map[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ] += 1

    probability_map /= np.maximum(
        count_map,
        1
    )

    return probability_map


def detect_components(binary_mask):

    labels, num_labels = ndimage.label(
        binary_mask
    )

    defects = []

    for c in range(1, num_labels + 1):

        ys, xs = np.where(labels == c)

        area = len(xs)

        if area < MIN_AREA:
            continue

        x = int(xs.min())
        y = int(ys.min())

        width = int(
            xs.max() - xs.min() + 1
        )

        height = int(
            ys.max() - ys.min() + 1
        )

        defects.append({
            "area": area,
            "x": x,
            "y": y,
            "width": width,
            "height": height
        })

    return defects, num_labels


def merge_close_components(
    defects,
    distance=20
):

    merged = []

    for d in sorted(
        defects,
        key=lambda x: x["area"],
        reverse=True
    ):

        x1 = d["x"]
        y1 = d["y"]
        x2 = x1 + d["width"]
        y2 = y1 + d["height"]

        merged_with_existing = False

        for m in merged:

            mx1 = m["x"]
            my1 = m["y"]
            mx2 = mx1 + m["width"]
            my2 = my1 + m["height"]

            close_x = (
                x1 <= mx2 + distance
                and
                x2 >= mx1 - distance
            )

            close_y = (
                y1 <= my2 + distance
                and
                y2 >= my1 - distance
            )

            if close_x and close_y:

                new_x1 = min(x1, mx1)
                new_y1 = min(y1, my1)

                new_x2 = max(x2, mx2)
                new_y2 = max(y2, my2)

                m["x"] = new_x1
                m["y"] = new_y1

                m["width"] = (
                    new_x2 - new_x1
                )

                m["height"] = (
                    new_y2 - new_y1
                )

                m["area"] += d["area"]

                merged_with_existing = True

                break

        if not merged_with_existing:

            merged.append(d.copy())

    return merged


def get_severity(area_percent):

    if area_percent >= 1.0:
        return "HIGH"

    elif area_percent >= 0.2:
        return "MEDIUM"

    else:
        return "LOW"


def create_report(defects, total_pixels, probability_map):

    rows = []

    for i, d in enumerate(defects, 1):

        area_percent = (
            d["area"] / total_pixels
        ) * 100

        x1 = d["x"]
        y1 = d["y"]
        x2 = x1 + d["width"]
        y2 = y1 + d["height"]

        region = probability_map[y1:y2, x1:x2]

        if region.size > 0:
            confidence = float(region.mean() * 100)
        else:
            confidence = 0.0

        severity = get_severity(area_percent)

        rows.append({
            "Defect_ID": f"D{i:02d}",
            "Area_pixels": d["area"],
            "X": d["x"],
            "Y": d["y"],
            "Width": d["width"],
            "Height": d["height"],
            "Area_percent": round(area_percent, 3),
            "Confidence": round(confidence, 2),
            "Severity": severity
        })

    return pd.DataFrame(rows)

def create_visualization(
    image,
    defects
):

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    ax.imshow(image)

    total_pixels = (
        image.shape[0] *
        image.shape[1]
    )

    for i, d in enumerate(
        defects,
        1
    ):

        area_percent = (
            d["area"] /
            total_pixels
        ) * 100

        severity = get_severity(
            area_percent
        )

        rect = plt.Rectangle(
            (
                d["x"],
                d["y"]
            ),
            d["width"],
            d["height"],
            fill=False,
            linewidth=2
        )

        ax.add_patch(rect)

        ax.text(
            d["x"],
            max(
                d["y"] - 8,
                5
            ),
            f"D{i} ({severity})",
            fontsize=8,
            fontweight="bold"
        )

    ax.set_title(
        "PCB Defect Detection"
    )

    ax.axis("off")

    plt.tight_layout()

    return fig


# ============================================================
# MAIN PIPELINE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    full_img = np.array(image)

    H, W, _ = full_img.shape

    st.subheader(
        "Uploaded PCB Image"
    )

    st.image(
        full_img,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PATCH EXTRACTION
    # --------------------------------------------------------

    with st.spinner(
        "Extracting PCB patches..."
    ):

        patches, positions = (
            extract_patches(
                full_img
            )
        )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with st.spinner(
        "Running U-Net defect detection..."
    ):

        patches_input = (
            patches.astype(
                "float32"
            ) / 255.0
        )

        predictions = model.predict(
            patches_input,
            batch_size=4,
            verbose=0
        )

    # --------------------------------------------------------
    # PROBABILITY MAP
    # --------------------------------------------------------

    probability_map = (
        create_probability_map(
            predictions,
            positions,
            H,
            W
        )
    )

    # --------------------------------------------------------
    # BINARY MASK
    # --------------------------------------------------------

    binary_mask = (
        probability_map >= THRESHOLD
    ).astype(np.uint8)

    defect_pixels = int(
        np.sum(binary_mask)
    )

    total_pixels = (
        H * W
    )

    defect_area = (
        defect_pixels /
        total_pixels
    ) * 100

    # --------------------------------------------------------
    # COMPONENT DETECTION
    # --------------------------------------------------------

    filtered_defects, raw_components = (
        detect_components(
            binary_mask
        )
    )

    # --------------------------------------------------------
    # MERGING
    # --------------------------------------------------------

    merged_defects = (
        merge_close_components(
            filtered_defects,
            MERGE_DISTANCE
        )
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report_df = create_report(
        merged_defects,
        total_pixels,
        probability_map
    )

    if len(report_df) > 0:

        severity_counts = (
            report_df["Severity"]
            .value_counts()
            .to_dict()
        )

    else:

        severity_counts = {}

    high_count = severity_counts.get(
        "HIGH",
        0
    )

    medium_count = severity_counts.get(
        "MEDIUM",
        0
    )

    low_count = severity_counts.get(
        "LOW",
        0
    )    # ========================================================
    # OVERALL PCB STATUS
    # ========================================================

    if len(report_df) == 0:

        overall_status = "PASS"

    else:

        total_predicted_area = report_df["Area_percent"].sum()

        if total_predicted_area >= 3:
            overall_status = "FAIL"
        else:
            overall_status = "REVIEW"
    #    # ========================================================
    # DASHBOARD
    # ========================================================

    st.subheader(
        "📊 PCB Inspection Summary"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Defect Area",
        f"{defect_area:.2f}%"
    )

    col2.metric(
        "Defect Groups",
        len(merged_defects)
    )

    col3.metric(
        "Medium",
        medium_count
    )

    col4.metric(
        "High",
        high_count
    )

    st.metric(
        "Low",
        low_count
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Confidence",
        f"{report_df['Confidence'].mean():.2f}%"
        if len(report_df) > 0
        else "0.00%"
    )

    col2.metric(
        "Total Defect Area",
        f"{report_df['Area_percent'].sum():.2f}%"
        if len(report_df) > 0
        else "0.00%"
    )

    col3.metric(
        "Overall Status",
        overall_status
    )
        # ========================================================
    # SEVERITY DISTRIBUTION
    # ========================================================

    if len(report_df) > 0:

        st.subheader(
            "📊 Severity Distribution"
        )

        severity_chart = (
            report_df["Severity"]
            .value_counts()
        )

        st.bar_chart(
            severity_chart
        )
    # ========================================================
    # DETECTED IMAGE
    # ========================================================

    st.subheader(
        "🔍 Detected Defects"
    )

    fig = create_visualization(
        full_img,
        merged_defects
    )

    st.pyplot(
        fig,
        use_container_width=True
    )
    # ========================================================
    # DEFECT TABLE
    # ========================================================

    st.subheader(
        "📋 Defect Details"
    )

    if len(report_df) > 0:

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

        #    # ----------------------------------------------------
    # CSV DOWNLOAD
    # ----------------------------------------------------

    csv_data = report_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Defect Report CSV",
        data=csv_data,
        file_name="PCB_Defect_Report.csv",
        mime="text/csv"
    )

else:

    st.success(
        "No defect regions detected."
    )

else:

    st.warning(
        "Please upload a PCB image to begin."
    )