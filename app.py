import streamlit as st
from PIL import Image

from src.gradcam import generate_cam_overlay
from src.pixiv_downloader import (
    build_search_word,
    create_pixiv_api,
    download_illust,
    filter_illusts,
    fetch_refresh_token_cli,
    ranking_illusts,
    search_illusts,
)

st.title("Touhou(东方) Character Classifier")

tab_classifier, tab_pixiv = st.tabs(["Classifier", "Pixiv Downloader"])

with tab_classifier:
    uploaded = st.file_uploader("Upload an image(上传你的东方人物)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        img.save("temp.png")

        cam, orig, label = generate_cam_overlay("temp.png")

        st.write(f"Prediction: **{label}**")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(orig)
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.axis("off")
        st.pyplot(fig)

with tab_pixiv:
    st.subheader("Pixiv Downloader")
    st.caption("Use a Pixiv refresh token to authenticate. Ugoira (animated) items are skipped.")

    if "pixiv_refresh_token" not in st.session_state:
        st.session_state.pixiv_refresh_token = ""

    with st.expander("Get refresh token (Pixiv login)", expanded=False):
        st.caption("Login happens in a browser window for CAPTCHA/2FA. Credentials are not stored.")
        st.info("A browser window will open for Pixiv login and CAPTCHA.")
        if st.button("Fetch refresh token", type="secondary"):
            try:
                st.session_state.pixiv_refresh_token = fetch_refresh_token_cli(interactive=True)
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

    col_ai, col_not_ai = st.columns(2)
    with col_ai:
        include_ai = st.checkbox("AI tagged", value=True)
    with col_not_ai:
        include_not_ai = st.checkbox("Not AI tagged", value=True)

    if mode == "Search by tag":
        word = st.text_input("Tag(s)", placeholder="e.g. 東方Project, 博麗霊夢")
        search_target = st.selectbox(
            "Search target",
            ["Partial Match Tags", "Exact Match Tags", "Title and Caption"],
        )
        sort = st.selectbox(
            "Sort",
            ["Date_desc", "Date_asc", "Popular_desc"],
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
            if not include_ai and not include_not_ai:
                st.error("Select at least one AI filter option.")
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

            search_ai_type = None
            if include_ai and not include_not_ai:
                search_ai_type = 2
            elif include_not_ai and not include_ai:
                search_ai_type = 0
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
            if not include_ai and not include_not_ai:
                st.error("Select at least one AI filter option.")
                st.stop()
            if include_ai and not include_not_ai:
                illusts = [illust for illust in illusts if getattr(illust, "ai_type", 0) == 2]
            elif include_not_ai and not include_ai:
                illusts = [illust for illust in illusts if getattr(illust, "ai_type", 0) == 0]
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



