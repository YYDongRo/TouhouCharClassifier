import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from src.dataset import TouhouImageDataset
from src.gradcam import generate_cam_overlay, load_model as load_gradcam_model
from src.memory import count_memory_entries, find_best_matches, save_memory_example
from src.pixiv_downloader import (
    build_search_word,
    create_pixiv_api,
    download_illust,
    filter_illusts,
    fetch_refresh_token,
    ranking_illusts,
    search_illusts,
)

st.title("Touhou(东方) Character Classifier")


@st.cache_data
def load_class_names() -> list[str]:
    class_map_path = Path("class_map.json")
    if class_map_path.exists():
        with class_map_path.open("r", encoding="utf-8") as f:
            class_map = json.load(f)
        idx_to_class = {int(v): k for k, v in class_map.items()}
        return [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    ds = TouhouImageDataset("data")
    return [ds.idx_to_class[i] for i in sorted(ds.idx_to_class.keys())]


@st.cache_resource
def load_memory_model():
    model, _idx_to_class, _class_to_idx = load_gradcam_model()
    return model


def resize_for_canvas(image: Image.Image, max_size: int = 640) -> tuple[Image.Image, float, float]:
    w, h = image.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = image.resize((new_w, new_h), Image.LANCZOS)
    else:
        new_w, new_h = w, h
        resized = image
    return resized, w / new_w, h / new_h


def extract_circle_region(
    image: Image.Image,
    circle_obj: dict,
    scale_x: float,
    scale_y: float,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = image.size
    left = float(circle_obj.get("left", 0)) * scale_x
    top = float(circle_obj.get("top", 0)) * scale_y
    radius = float(circle_obj.get("radius", 0))
    sx = float(circle_obj.get("scaleX", 1))
    sy = float(circle_obj.get("scaleY", 1))
    rx = radius * sx * scale_x
    ry = radius * sy * scale_y
    r = max(rx, ry)
    cx = left + rx
    cy = top + ry

    x0 = int(max(cx - r, 0))
    y0 = int(max(cy - r, 0))
    x1 = int(min(cx + r, w))
    y1 = int(min(cy + r, h))
    bbox = (x0, y0, x1, y1)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    crop = image.crop(bbox)
    mask_crop = mask.crop(bbox)
    masked = Image.composite(crop, Image.new("RGB", crop.size, (0, 0, 0)), mask_crop)
    return masked, bbox

tab_classifier, tab_pixiv = st.tabs(["Classifier", "Pixiv Downloader"])

with tab_classifier:
    uploaded = st.file_uploader("Upload an image(上传你的东方人物)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        img.save("temp.png")

        cam, orig, label, probs = generate_cam_overlay("temp.png")

        # Check confidence threshold
        CONFIDENCE_THRESHOLD = 0.45
        top_prob = probs[0][1] if probs else 0.0
        
        if top_prob < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ **Low confidence prediction** ({top_prob:.1%})\n\n"
                "The AI cannot confidently recognize this character. Possible reasons:\n"
                "- This character is not in the training data\n"
                "- The image quality is low\n"
                "- Multiple characters in the image\n\n"
                "Consider training more classes or adding this character to the dataset."
            )
        
        st.write(f"Prediction: **{label}**")

        st.subheader("Class probabilities")
        prob_rows = [{"Class": cls, "Probability": float(p)} for cls, p in probs]
        st.dataframe(prob_rows, use_container_width=True)

        class_names = [cls for cls, _ in probs]
        target = st.selectbox(
            "Grad-CAM target class",
            options=class_names,
            index=0,
            help="Select a class to visualize its activation map.",
        )

        if target != label:
            cam, orig, label, probs = generate_cam_overlay("temp.png", target_class=target)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(orig)
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.axis("off")
        st.pyplot(fig)

        st.subheader("Teach / Correct (Circle memory)")
        with st.expander("Help the app memorize a character", expanded=False):
            st.caption(
                "Circle the character, choose the correct label, and save. "
                "These examples are stored locally and used for similarity matching."
            )

            class_names_all = load_class_names()
            default_idx = class_names_all.index(label) if label in class_names_all else 0
            correct_label = st.selectbox("Correct label", options=class_names_all, index=default_idx)
            note = st.text_input("Note (optional)", placeholder="e.g., headshot, chibi, side view")
            use_circle_only = st.checkbox("Use circled region only", value=True)

            resized_img, scale_x, scale_y = resize_for_canvas(img)
            circle_obj = None

            cx = st.slider("Circle center X", 0, resized_img.width, resized_img.width // 2)
            cy = st.slider("Circle center Y", 0, resized_img.height, resized_img.height // 2)
            max_r = max(5, min(resized_img.width, resized_img.height) // 2)
            r = st.slider("Circle radius", 5, max_r, max_r // 2)
            circle_obj = {
                "left": cx - r,
                "top": cy - r,
                "radius": r,
                "scaleX": 1.0,
                "scaleY": 1.0,
            }
            preview = resized_img.copy()
            draw = ImageDraw.Draw(preview)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 75, 75), width=3)
            st.image(preview, caption="Circle preview")

            if st.button("Save to memory", type="primary"):
                if use_circle_only and not circle_obj:
                    st.error("Please draw a circle around the character.")
                else:
                    if circle_obj:
                        region, bbox = extract_circle_region(img, circle_obj, scale_x, scale_y)
                    else:
                        region, bbox = img, None
                    model = load_memory_model()
                    saved_path = save_memory_example(
                        model,
                        region,
                        correct_label,
                        source_path=str(getattr(uploaded, "name", "uploaded")),
                        bbox=bbox,
                        note=note,
                    )
                    st.success(f"Saved to memory: {saved_path}")

            st.caption(f"Memory examples: {count_memory_entries()}")

            if st.checkbox("Show memory matches"):
                query_image = img
                if use_circle_only and circle_obj:
                    query_image, _bbox = extract_circle_region(img, circle_obj, scale_x, scale_y)
                model = load_memory_model()
                matches = find_best_matches(model, query_image, top_k=5)
                if matches:
                    st.dataframe(matches, use_container_width=True)
                    best = matches[0]
                    st.info(
                        f"Best memory match: {best['label']} (score {best['score']:.3f}). "
                        "You can save a correction above if needed."
                    )
                else:
                    st.info("No memory examples yet. Save one above to start.")

with tab_pixiv:
    st.subheader("Pixiv Downloader")

    if "pixiv_refresh_token" not in st.session_state:
        st.session_state.pixiv_refresh_token = ""

    # Login mode selection
    login_mode = st.radio(
        "Login mode",
        ["Headless (auto)", "Visible browser (auto)"],
        horizontal=True,
        help="Headless runs in background. Visible browser shows the login process."
    )

    if st.button("Fetch refresh token", type="secondary"):
        try:
            if login_mode == "Headless (auto)":
                st.session_state.pixiv_refresh_token = fetch_refresh_token(headless=True, manual_login=False)
            else:
                st.session_state.pixiv_refresh_token = fetch_refresh_token(headless=False, manual_login=False)
            
            st.success("Refresh token fetched.")
        except Exception as exc:
            st.error(f"Failed to fetch refresh token: {exc}")

    refresh_token = st.text_input(
        "Refresh token",
        type="password",
        value=st.session_state.pixiv_refresh_token,
    )

    mode = st.radio("Mode", ["Search by tag", "Ranking"], horizontal=True)

    search_target = "partial_match_for_tags"
    sort = "date_desc"
    duration = None
    word = ""
    ranking_mode = "day"
    ranking_date = ""

    age_filter = st.selectbox("Age filter", ["All ages", "R-18"])

    include_ai = st.checkbox("Include AI artwork", value=False)

    if mode == "Search by tag":
        word = st.text_input("Tag(s)", placeholder="e.g. 東方Project, 博麗霊夢")
        search_target = st.selectbox(
            "Search target",
            ["Partial Match Tags", "Exact Match Tags", "Title and Caption"],
        )
        sort = st.selectbox(
            "Sort",
            ["Popular_desc", "Date_asc", "Date_desc"],
            help="Popular_desc may require Pixiv Premium.",
        )
        duration = st.selectbox(
            "Duration filter",
            ["All Period", "Within last day", "Within last week", "Within last month"],
        )
        duration = None if duration == "All Period" else duration
    else:
        base_ranking_mode = st.selectbox(
            "Ranking mode",
            ["day", "week", "month"],
        )
        ranking_date = st.text_input("Ranking date (YYYY-MM-DD, optional)")
        ranking_mode = f"{base_ranking_mode}_r18" if age_filter == "R-18" else base_ranking_mode

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        max_items = st.number_input("Max items", min_value=1, max_value=200, value=30)
    with col_b:
        max_pages = st.number_input("Max pages", min_value=1, max_value=20, value=5)
    with col_c:
        size = st.selectbox("Image size", ["Original", "Large", "Medium"])

    first_only = st.checkbox("Only download first image", value=False)

    output_dir = st.text_input("Output folder", value="data/name")

    if st.button("Download", type="primary"):
        if not refresh_token:
            st.error("Refresh token is required.")
            st.stop()

        try:
            aapi = create_pixiv_api(refresh_token)
        except Exception as exc:
            st.error(f"Failed to authenticate: {exc}")
            st.stop()

        if mode == "Search by tag":
            include_r18 = age_filter == "R-18"
            search_word = build_search_word(word, include_r18)
            if not search_word:
                st.error("Please enter at least one tag.")
                st.stop()

            search_target_map = {
                "Partial Match Tags": "partial_match_for_tags",
                "Exact Match Tags": "exact_match_for_tags",
                "Title and Caption": "title_and_caption",
            }
            sort_map = {
                "Date_desc": "date_desc",
                "Date_asc": "date_asc",
                "Popular_desc": "popular_desc",
            }
            duration_map = {
                "Within last day": "within_last_day",
                "Within last week": "within_last_week",
                "Within last month": "within_last_month",
            }

            search_target_api = search_target_map.get(search_target, "partial_match_for_tags")
            sort_api = sort_map.get(sort, "date_desc")
            duration_api = duration_map.get(duration)

            # search_ai_type: None = all, 0 = not AI, 2 = AI only
            search_ai_type = None if include_ai else 0
            illusts = search_illusts(
                aapi,
                search_word,
                search_target_api,
                sort_api,
                duration_api,
                search_ai_type,
                max_items,
                max_pages,
            )
            count_api = len(illusts)
            count_ai = count_api
        else:
            illusts = ranking_illusts(
                aapi,
                ranking_mode,
                ranking_date.strip() or None,
                max_items,
                max_pages,
            )
            count_api = len(illusts)
            if not include_ai:
                illusts = [illust for illust in illusts if getattr(illust, "ai_type", 0) != 2]
            count_ai = len(illusts)

        illusts = filter_illusts(illusts)
        count_type = len(illusts)

        st.caption(
            f"Fetched {count_api} items from API; {count_ai} after AI filter; {count_type} after excluding manga/novel/ugoira."
        )

        if not illusts:
            st.warning("No results found.")
            st.stop()

        progress = st.progress(0)
        downloaded_paths = []
        failed_downloads = []
        for idx, illust in enumerate(illusts):
            try:
                size_map = {"Original": "original", "Large": "large", "Medium": "medium"}
                size_api = size_map.get(size, "original")
                paths = download_illust(aapi, illust, output_dir, size=size_api, first_only=first_only)
                if not paths:
                    failed_downloads.append((getattr(illust, "id", "unknown"), "no downloadable image"))
                else:
                    downloaded_paths.extend(paths)
            except Exception as exc:
                failed_downloads.append((getattr(illust, "id", "unknown"), str(exc)))
            progress.progress((idx + 1) / len(illusts))

        st.success(f"Downloaded {len(downloaded_paths)} image(s) to {output_dir}.")
        if failed_downloads:
            st.warning(f"Failed to download {len(failed_downloads)} work(s).")
            with st.expander("See failed items"):
                for illust_id, reason in failed_downloads[:50]:
                    st.write(f"{illust_id}: {reason}")



