import os
import argparse
import logging
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import streamlit.components.v1 as components


st.set_page_config(layout="wide", page_title="Similarity UI", initial_sidebar_state="collapsed")

RATING_SCALE = list(range(10))  # 0 to 9

st.markdown(
    """
    <style>
      .block-container {max-width: 2000px; padding-left: 1rem; padding-right: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ -------------------------

parser = argparse.ArgumentParser(description="Validation UI for pairwise similarity ratings")
parser.add_argument("data_dir", nargs="?", default="data", help="Папка с данными")
parser.add_argument("requests_file", nargs="?", default="requests.parquet", help="Файл с запросами (.parquet или .csv)")
parser.add_argument("pairs_file", nargs="?", default="similarities.csv", help="Файл с парами (.csv)")
parser.add_argument("wrap_width", nargs="?", default=400, help="Ширина переноса текста")
args, _ = parser.parse_known_args()

DATA_DIR = args.data_dir
REQUESTS_FILE = args.requests_file
SIMILARITIES_FILE = args.pairs_file
WRAP_WIDTH = args.wrap_width

# ------------------------- КЭШИРУЕМАЯ ЗАГРУЗКА REQUESTS -------------------------
@st.cache_data(show_spinner=False)
def load_requests(path, mtime):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def build_index(df):
    return df.set_index("id")[["subject", "cleaned_request", "details"]]

req_path = os.path.join(DATA_DIR, REQUESTS_FILE)
req_mtime = os.path.getmtime(req_path)
df_req = load_requests(req_path, req_mtime)
req_by_id = build_index(df_req)

# ------------------------- ЗАГРУЗКА ПАР (БЕЗ КЭША) -------------------------
pairs_path = os.path.join(DATA_DIR, SIMILARITIES_FILE)
df_pairs = pd.read_csv(pairs_path, dtype={"score": str})
# Оставляем score строковым ("похожи"/"не похожи"); если колонки нет — создаём пустую
if "score" not in df_pairs.columns:
    df_pairs["score"] = ""

# ------------------------- ДИСКЛЕЙМЕРЫ И ПОДПИСИ -------------------------
disclaimers = {
    "name": "disclaimers",
    "label": "Запрос плохо выделен или отсутствует в исходном файле",
    "path": os.path.join(DATA_DIR, "bad_requests.csv"),
    "df": None}

signatures = {
    "name": "signatures",
    "label": "Замечена неудалённая подпись или дисклеймер",
    "path": os.path.join(DATA_DIR, "suspected_signatures.csv"),
    "df": None}


def init_checkboxes(d):
    path = d["path"]
    if os.path.exists(path):
        df = pd.read_csv(path).set_index("id")
    else:
        ids = pd.unique(pd.concat([df_pairs["id1"].astype(str), df_pairs["id2"].astype(str)]))
        df = pd.DataFrame({"id": ids, "detected": 0}).set_index("id")
        df.reset_index().to_csv(path, index=False)
    return df


def save_checkbox(d,id):
    d["df"].at[id, "detected"] = 1 if st.session_state.get(f"{d['name']}_{id}", False) else 0
    d["df"].reset_index().to_csv(d["path"], index=False)


disclaimers["df"] = init_checkboxes(disclaimers)
signatures["df"] = init_checkboxes(signatures)

# ------------------------- СОСТОЯНИЕ КУРСОРА -------------------------
if "i" not in st.session_state:
    st.session_state.i = 0
st.session_state.i = max(0, min(st.session_state.i, len(df_pairs) - 1))

# Версия для перерисовки AgGrid (сброс выделения при смене строки программно)
if "grid_version" not in st.session_state:
    st.session_state.grid_version = 0

row = df_pairs.iloc[st.session_state.i]
id1, id2 = str(row["id1"]), str(row["id2"])
_raw_score = row.get("score") if isinstance(row, pd.Series) else None
score_label = None
if _raw_score is not None and not pd.isna(_raw_score) and str(_raw_score).strip() != "":
    score_label = str(_raw_score).strip()

left = req_by_id.loc[id1]
right = req_by_id.loc[id2]

def wrap(text, width):
    return "\n".join(text[i:i+width] for i in range(0, len(text), width))

clicked = None

# ------------------------- ДВЕ КОЛОНКИ С ТЕКСТАМИ -------------------------

st.markdown("""
    <style>
        /* Ensure text areas use black text everywhere, including disabled */
        .stTextArea textarea,
        .stTextArea div[contenteditable="true"],
        .stTextArea div[data-baseweb="textarea"] textarea {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important; /* Safari */
            caret-color: #000000 !important;
        }
        .stTextArea textarea:disabled,
        .stTextArea div[data-baseweb="textarea"] textarea:disabled {
            opacity: 1 !important; /* prevent greyed-out look */
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }
        .stTextArea textarea::placeholder {
            color: #000000 !important;
            opacity: 0.6 !important;
        }
        /* Thin border around the right main column */
        .main-col-box {
            border: 1px solid #d6d6d6;
            border-radius: 6px;
            padding: 12px 16px;
        }
    </style>
""", unsafe_allow_html=True)

side_col, main_col = st.columns([1, 4], vertical_alignment="top")

with main_col:

    st.markdown("**Оцените похожесть запросов, нажав соответствующую кнопку:**")

    current_score = df_pairs.at[df_pairs.index[st.session_state.i], "score"] if "score" in df_pairs.columns else ""
    try:
        current_score_int = int(current_score) if current_score and str(current_score).strip() else -1
    except (ValueError, TypeError):
        current_score_int = -1
    
    cols = st.columns(10)
    
    for i in range(10):
        with cols[i]:
            button_type = "primary" if i == current_score_int else "secondary"
            if st.button(
                str(i),
                key=f"rating_btn_{i}_{st.session_state.i}",
                use_container_width=True,
                type=button_type
            ):
                clicked = str(i)
    
    st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown("**Оценки похожести**")
    display_cols = [c for c in ["id1", "id2", "score"] if c in df_pairs.columns]
    if len(df_pairs) > 0:
        # AgGrid: кликаем по строке — получаем выбранную строку
        table_df = df_pairs[display_cols].copy().reset_index(drop=True)
        table_df.insert(0, "row_idx", range(len(table_df)))  # скрытый индекс для маппинга
        gb = GridOptionsBuilder.from_dataframe(table_df)
        gb.configure_column("id1", width=50)
        gb.configure_column("id2", width=50)
        gb.configure_column("score", width=30)

        # Включаем выбор строки по клику и предварительно выделяем текущую пару
        pre_idx = min(max(int(st.session_state.i), 0), len(table_df) - 1) if len(table_df) else 0
        gb.configure_selection(
            selection_mode='single',
            use_checkbox=False,
            rowMultiSelectWithClick=True,
            pre_selected_rows=[pre_idx],
        )
        gb.configure_column("row_idx", hide=True)
        # Подсветка выбранной и текущей строк
        row_style_js = JsCode(
            f"""
            function(params) {{
                // Голубая подсветка выбранной строки
                if (params.node && params.node.isSelected && params.node.isSelected()) {{
                    return {{ backgroundColor: '#cce5ff' }};
                }}
                // Жёлтая подсветка текущей строки (по индексу)
                if (params.data && params.data.row_idx === {pre_idx}) {{
                    return {{ backgroundColor: '#fff3cd' }};
                }}
                return null;
            }}
            """
        )
        # Автопрокрутка к текущей записи при отрисовке и при выборе
        on_first_render_js = JsCode(
            f"""
            function(params) {{
                var api = params.api;
                if (api && typeof api.ensureIndexVisible === 'function') {{
                    api.ensureIndexVisible({pre_idx}, 'middle');
                }}
            }}
            """
        )
        gb.configure_grid_options(
            domLayout='normal',
            suppressRowClickSelection=False,
            getRowStyle=row_style_js,
            onFirstDataRendered=on_first_render_js,
        )
        grid_options = gb.build()

        grid_resp = AgGrid(
            table_df,
            gridOptions=grid_options,
            height=600,
            fit_columns_on_grid_load=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
            reload_data=False,
            theme='streamlit',
            key=f"pairs_aggrid_{st.session_state.grid_version}",
        )

        # Извлекаем выбранные строки из ответа AgGrid
        sel_rows_raw = None
        if isinstance(grid_resp, dict):
            sel_rows_raw = grid_resp.get("selected_rows", None)
        else:
            sel_rows_raw = getattr(grid_resp, "selected_rows", None)

        # альтернативный ключ, если используется другая версия
        if sel_rows_raw is None:
            if isinstance(grid_resp, dict):
                sel_rows_raw = grid_resp.get("selectedRows", None)
            else:
                sel_rows_raw = getattr(grid_resp, "selectedRows", None)

        if isinstance(sel_rows_raw, pd.DataFrame):
            sel_rows = sel_rows_raw.to_dict("records")
        elif isinstance(sel_rows_raw, list):
            sel_rows = sel_rows_raw
        elif sel_rows_raw is None:
            sel_rows = []
        else:
            # неожиданный тип — пробуем привести к списку
            try:
                sel_rows = list(sel_rows_raw)
            except Exception:
                logging.exception(
                    "AgGrid: unexpected type for selected rows; type=%s value=%r",
                    type(sel_rows_raw), sel_rows_raw,
                )
                sel_rows = []

        # Если ничего не выбрано, просто ждём клика (предварительная подсветка уже включена)

        if sel_rows:
            try:
                new_idx = int(sel_rows[0].get("row_idx"))
                if 0 <= new_idx < len(df_pairs) and new_idx != st.session_state.i:
                    st.session_state.i = new_idx
                    st.session_state.grid_version += 1
                    st.rerun()
            except Exception:
                logging.exception("AgGrid: failed to process selected row; sel_rows=%r", sel_rows)
    else:
        st.info("Нет пар для отображения.")

    total_pairs = len(df_pairs) if isinstance(df_pairs, pd.DataFrame) else 0
    if "score" in df_pairs.columns:
        def is_rated(score):
            try:
                return 1 <= int(score) <= 9
            except (ValueError, TypeError):
                return False
                
        total_scored = df_pairs["score"].apply(is_rated).sum()
        scores = pd.to_numeric(df_pairs["score"], errors='coerce')
        low_scores = ((scores >= 0) & (scores <= 3)).sum()
        medium_scores = ((scores >= 4) & (scores <= 6)).sum()
        high_scores = ((scores >= 7) & (scores <= 9)).sum()
    else:
        total_scored = low_scores = medium_scores = high_scores = 0
    remaining = max(total_pairs - total_scored, 0)

    st.markdown(
        f"Оценено: {total_scored} из {total_pairs} пар")

    if total_scored > 0:
        st.markdown(f"Распределение оценок:\n - низкие (1-3): {low_scores}\n - средние (4-6): {medium_scores}\n - высокие (7-9): {high_scores}")

    st.markdown(f"Осталось оценить: {remaining} из {total_pairs}")

    progress_ratio = (total_scored / total_pairs) if total_pairs > 0 else 0.0
    st.progress(progress_ratio)

    try:
        main_box = main_col.container(border=True)  # Streamlit >= 1.31
    except TypeError:
        main_box = main_col.container()  # Fallback without border
    with main_box:
        # Two columns with short texts
        c1, c2 = st.columns(2, vertical_alignment="top")
        # --- НОВЫЙ РЯД: чекбоксы в одну линию (2 для левой записи, 2 для правой) ---
        # cb_l_disc, cb_l_sign, spacer, cb_r_disc, cb_r_sign = st.columns(
        #     [1, 1, 0.2, 1, 1],  # пропорции можно подстроить
        #     vertical_alignment="center")
        cb_l_disc, cb_l_sign, cb_r_disc, cb_r_sign = st.columns(4, vertical_alignment="center")
        # Two columns for full file content
        fm1, fm2 = st.columns(2)

with c1:
    st.markdown(f"#### {id1}")
    subject = left["subject"]
    if isinstance(subject, pd.Series):
        subject_text = str(subject.iloc[0]) if not subject.empty and not pd.isna(subject.iloc[0]) else ""
    else:
        subject_text = str(subject) if not pd.isna(subject) else ""
    st.text_area(
        "Subject",
        wrap(subject_text, WRAP_WIDTH),
        height=40,
        key=f"subj_{id1}",
        disabled=True
    )
    st.text_area(
        "Очищенный запрос",
        wrap(str(left["cleaned_request"]), WRAP_WIDTH),
        height=140,
        key=f"ta_{id1}",
        disabled=True
    )


with c2:
    st.markdown(f"#### {id2}")
    subject = right["subject"]
    if isinstance(subject, pd.Series):
        subject_text = str(subject.iloc[0]) if not subject.empty and not pd.isna(subject.iloc[0]) else ""
    else:
        subject_text = str(subject) if not pd.isna(subject) else ""
    st.text_area(
        "Subject",
        wrap(subject_text, WRAP_WIDTH),
        height=40,
        key=f"subj_{id2}",
        disabled=True
    )
    st.text_area(
        "Очищенный запрос",
        wrap(str(right["cleaned_request"]), WRAP_WIDTH),
        height=140,
        key=f"ta_{id2}",
        disabled=True
    )


with cb_l_disc:
    st.checkbox(
        disclaimers["label"],
        value=bool(disclaimers["df"].at[id1, "detected"]),
        key=f"{disclaimers['name']}_{id1}",
        on_change=save_checkbox,
        args=(disclaimers, id1),
    )

with cb_l_sign:
    st.checkbox(
        signatures["label"],
        value=bool(signatures["df"].at[id1, "detected"]),
        key=f"{signatures['name']}_{id1}",
        on_change=save_checkbox,
        args=(signatures, id1),
    )

with cb_r_disc:
    st.checkbox(
        disclaimers["label"],
        value=bool(disclaimers["df"].at[id2, "detected"]),
        key=f"{disclaimers['name']}_{id2}",
        on_change=save_checkbox,
        args=(disclaimers, id2),
    )

with cb_r_sign:
    st.checkbox(
        signatures["label"],
        value=bool(signatures["df"].at[id2, "detected"]),
        key=f"{signatures['name']}_{id2}",
        on_change=save_checkbox,
        args=(signatures, id2),
    )



with fm1:
    st.markdown("---")
    if "detals" in df_req.columns:
        _series = df_req.loc[df_req["id"].astype(str) == id1, "detals"]
        full_left = str(_series.iloc[0]) if not _series.empty and not pd.isna(_series.iloc[0]) else ""
    else:
        val = left.get("details") if isinstance(left, pd.Series) else None
        full_left = str(val) if val is not None and not pd.isna(val) else ""
    st.text_area(
        "Файл полностью",
        wrap(full_left, WRAP_WIDTH),
        height=600,
        key=f"full_{id1}",
        disabled=True,
    )

with fm2:
    st.markdown("---")
    if "detals" in df_req.columns:
        _series = df_req.loc[df_req["id"].astype(str) == id2, "detals"]
        full_right = str(_series.iloc[0]) if not _series.empty and not pd.isna(_series.iloc[0]) else ""
    else:
        val = right.get("details") if isinstance(right, pd.Series) else None
        full_right = str(val) if val is not None and not pd.isna(val) else ""
    st.text_area(
        "Файл полностью",
        wrap(full_right, WRAP_WIDTH),
        height=600,
        key=f"full_{id2}",
        disabled=True,
    )

 

# ------------------------- СОХРАНЕНИЕ ОЦЕНКИ -------------------------

if clicked and clicked.isdigit() and 0 <= int(clicked) <= 9:  # сохраняем числовую оценку в score
    df_pairs.at[df_pairs.index[st.session_state.i], "score"] = clicked
    df_pairs.to_csv(pairs_path, index=False)  # пишем весь CSV

    # Найти следующую НЕоценённую пару: score не является числом от 1 до 9
    score_series = df_pairs.get("score")
    if score_series is not None:
        def is_valid_rating(x):
            if pd.isna(x):
                return True
            try:
                return not (1 <= int(x) <= 9)
            except (ValueError, TypeError):
                return True
                
        mask_unscored = score_series.apply(is_valid_rating)
        unscored_indices = list(df_pairs.index[mask_unscored])
        if unscored_indices:
            # Ищем первую после текущей, иначе первую вообще
            cur_pos = st.session_state.i
            # df_pairs.index может быть не 0..N-1, поэтому работаем с позицией
            # получим позиции всех неоцененных
            unscored_positions = [df_pairs.index.get_loc(idx) for idx in unscored_indices]
            next_pos_after = [pos for pos in unscored_positions if pos > cur_pos]
            if next_pos_after:
                st.session_state.i = min(next_pos_after)
                st.session_state.grid_version += 1
                st.rerun()
            else:
                st.session_state.i = min(unscored_positions)
                st.session_state.grid_version += 1
                st.rerun()
        else:
            st.success("Оценка сохранена! Неоценённых пар больше нет.")
            import time
            time.sleep(0.5)
    else:
        # На всякий случай, если колонки нет
        if st.session_state.i < len(df_pairs) - 1:
            st.session_state.i += 1
            st.session_state.grid_version += 1
            st.rerun()
        else:
            st.success("Оценка сохранена!")
            import time
            time.sleep(0.5)
