# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # ─────────────────────────────────────────────
# # KONFIGURASI HALAMAN
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="E-Commerce ML Dashboard",
#     page_icon="🛒",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ─────────────────────────────────────────────
# # LOAD DATA & MODEL (cached agar tidak reload terus)
# # ─────────────────────────────────────────────
# @st.cache_data
# def load_data():
#     df = pd.read_csv("c:proje\zeka\Data\Processed\cleaned_ecommerce_data.csv")
#     df["order_date"] = pd.to_datetime(df["order_date"])
#     return df

# @st.cache_resource
# def load_models():
#     models = {}
#     model_files = {
#         "trending":        "Notebooks/model_trending.pkl",
#         "risk":            "Notebooks/risk_model.pkl",
#         "recommendation":  "Notebooks/model_recommendation.pkl",
#         "forecast":        "Notebooks/forecast_model.pkl",
#         "le_category":     "Notebooks/le_category.pkl",
#         "le_product":      "Notebooks/le_product.pkl",
#         "le_region":       "Notebooks/le_region.pkl",
#         "encoders":        "Notebooks/encoders.pkl",
#     }
#     for name, path in model_files.items():
#         try:
#             with open(path, "rb") as f:
#                 models[name] = pickle.load(f)
#         except FileNotFoundError:
#             models[name] = None
#     return models

# # # Load semua
# # try:
# #     df = load_data()
# #     models = load_models()
# #     data_loaded = True
# # except Exception as e:
# #     data_loaded = False
# #     st.error(f"❌ Gagal memuat data: {e}")
# #     st.stop()

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import joblib 


st.set_page_config(
    page_title="E-Commerce ML Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GANTI PATH INI sesuai lokasi file CSV kamu ──
CSV_PATH  = r"C:\PROJE\zeka\Data\Processed\cleaned_ecommerce_data.csv"
MODEL_DIR = r"C:\PROJE\zeka\Notebooks"
# ────────────────────────────────────────────────


@st.cache_data
def load_data(path):
    # Tambahkan sep=';' agar Pandas tahu pemisahnya adalah titik koma
    df = pd.read_csv(path, sep=';')
    
    # Sekarang kolom "order_date" pasti terbaca
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    
    return df

@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

#── Load data ──
if not os.path.exists(CSV_PATH):
    st.error(f"❌ File tidak ditemukan:\n`{CSV_PATH}`")
else:
    df = load_data(CSV_PATH)


# ── Load models ──
models = {
    "trending":       load_model(os.path.join(MODEL_DIR, "model_trending.pkl")),
    "risk":           load_model(os.path.join(MODEL_DIR, "risk_model.pkl")),
    "recommendation": load_model(os.path.join(MODEL_DIR, "model_recommendation.pkl")),
    "forecast":       load_model(os.path.join(MODEL_DIR, "forecast_model.pkl")),
    "le_category":    load_model(os.path.join(MODEL_DIR, "le_category.pkl")),
    "le_product":     load_model(os.path.join(MODEL_DIR, "le_product.pkl")),
    "le_region":      load_model(os.path.join(MODEL_DIR, "le_region.pkl")),
    "encoders":       load_model(os.path.join(MODEL_DIR, "encoders.pkl")),
}

# ─────────────────────────────────────────────
# SIDEBAR NAVIGASI & FILTER
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 ML Dashboard")
    st.markdown("---")

    page = st.radio("📌 Halaman", [
        "📊 EDA Overview",
        "📈 Trending Analysis",
        "⚠️ Risk Prediction",
        "🎯 Recommendation",
        "🗺️ Region Analysis",
    ])

    st.markdown("---")
    st.subheader("🔧 Filter Global")

    # Filter tanggal
    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()
    date_range = st.date_input(
        "Rentang Tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter region
    regions = ["Semua"] + sorted(df["region"].unique().tolist())
    selected_region = st.selectbox("Region", regions)

    # Filter products
    products = ["Semua"] + sorted(df["product_name"].unique().tolist())
    selected_product = st.selectbox("Pilih Produk", products)

    st.markdown("---")
    st.caption(f"Total data: {len(df):,} baris")

# Terapkan filter
filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["order_date"].dt.date >= date_range[0]) &
        (filtered_df["order_date"].dt.date <= date_range[1])
    ]
if selected_region != "Semua":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]
if selected_product != "Semua":
    filtered_df = filtered_df[filtered_df["product_name"] == selected_product]

# ─────────────────────────────────────────────
# HALAMAN 1: EDA OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 EDA Overview":
    st.title("📊 EDA Overview")
    st.caption(f"Menampilkan {len(filtered_df):,} dari {len(df):,} transaksi")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Penjualan", f"Rp {filtered_df['sales'].sum():,.0f}")
    with col2:
        st.metric("Total Profit", f"Rp {filtered_df['profit'].sum():,.0f}")
    with col3:
        st.metric("Jumlah Transaksi", f"{len(filtered_df):,}")
    with col4:
        avg_discount = filtered_df["discount"].mean() * 100 if "discount" in filtered_df.columns else 0
        st.metric("Rata-rata Diskon", f"{avg_discount:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Grafik penjualan harian
    with col_left:
        st.subheader("Tren Penjualan Harian")
        sales_daily = (
            filtered_df.groupby(filtered_df["order_date"].dt.to_period("M"))["sales"]
            .sum()
            .reset_index()
        )
        sales_daily["order_date"] = sales_daily["order_date"].astype(str)
        fig = px.line(
            sales_daily, x="order_date", y="sales",
            labels={"order_date": "Bulan", "sales": "Total Penjualan"},
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Penjualan per kategori
    with col_right:
        st.subheader("Penjualan per produk")
        cat_sales = (
            filtered_df.groupby("product_name")["sales"].sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig2 = px.bar(
            cat_sales, x="product_name", y="sales",
            color="sales", color_continuous_scale="Blues",
            labels={"product_name": "Produk", "sales": "Total Penjualan"}
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Distribusi profit
    col3a, col3b = st.columns(2)
    with col3a:
        st.subheader("Distribusi Profit")
        fig3 = px.histogram(
            filtered_df, x="profit", nbins=40,
            color_discrete_sequence=["#00CC96"],
            labels={"profit": "Profit"}
        )
        fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with col3b:
        st.subheader("Top 10 Produk")
        top_products = (
            filtered_df.groupby("product_name")["sales"].sum()
            .nlargest(10)
            .reset_index()
        )
        fig4 = px.bar(
            top_products.sort_values("sales"), x="sales", y="product_name",
            orientation="h", color_discrete_sequence=["#AB63FA"],
            labels={"product_name": "Produk", "sales": "Penjualan"}
        )
        fig4.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig4, use_container_width=True)

    # Tabel data mentah
    with st.expander("📋 Lihat Data Mentah"):
        st.dataframe(filtered_df.head(100), use_container_width=True)

# ─────────────────────────────────────────────
# HALAMAN 2: TRENDING ANALYSIS
# ─────────────────────────────────────────────
elif page == "📈 Trending Analysis":
    st.title("📈 Trending Analysis")

    try:
        model_trending = joblib.load("model_trending.pkl")
        models = {"trending": model_trending}
    except:
        models = {"trending": None}
        st.error("File model_trending.pkl tidak ditemukan!")


    if "is_trending" in filtered_df.columns:
        st.write("Daftar Kolom:", filtered_df.columns.tolist())
        col1, col2 = st.columns([1, 2])

        with col1:
            trending_count = filtered_df["is_trending"].value_counts()
            labels = ["Tidak Trending", "Trending"]
            values = [trending_count.get(0, 0), trending_count.get(1, 0)]

            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values,
                hole=0.45,
                marker_colors=["#EF553B", "#00CC96"]
            )])
            fig.update_layout(title="Proporsi Trending", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Produk Trending per region")
            trending_cat = (
                filtered_df[filtered_df["is_trending"] == 1]
                .groupby("region")["product_name"]
                .count()
                .reset_index()
                .rename(columns={"product_name": "jumlah_trending"})
                .sort_values("jumlah_trending", ascending=False)
            )
            fig2 = px.bar(
                trending_cat, x="category", y="jumlah_trending",
                color="jumlah_trending", color_continuous_scale="Viridis",
                labels={"category": "Kategori", "jumlah_trending": "Jumlah Trending"}
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Prediksi trending produk baru
    st.markdown("---")
    st.subheader("🔮 Prediksi Trending Produk Baru")

    if models.get("trending") is not None:
        with st.form("trending_form"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                qty = st.number_input("Quantity", min_value=1, value=10)
                price = st.number_input("Unit Price", min_value=0.0, value=50000.0)
            with col_b:
                discount = st.slider("Diskon (%)", 0, 50, 10) / 100
                sales_val = st.number_input("Sales", min_value=0.0, value=500000.0)
            with col_c:
                profit_val = st.number_input("Profit", value=50000.0)

            submitted = st.form_submit_button("Prediksi Trending ▶")

        if submitted:
            try:
                input_data = np.array([[qty, price, discount, sales_val, profit_val]])
                pred = models["trending"].predict(input_data)[0]
                prob = models["trending"].predict_proba(input_data)[0]

                if pred == 1:
                    st.success(f"✅ Produk ini **TRENDING** — Probabilitas: {prob[1]*100:.1f}%")
                else:
                    st.warning(f"⚠️ Produk ini **TIDAK trending** — Probabilitas trending: {prob[1]*100:.1f}%")
            except Exception as e:
                st.error(f"Prediksi gagal: {e}")

# ─────────────────────────────────────────────
# HALAMAN 3: RISK PREDICTION
# ─────────────────────────────────────────────
elif page == "⚠️ Risk Prediction":
    st.title("⚠️ Risk Prediction")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Profit per Region")
        fig = px.box(
            filtered_df, x="region", y="profit",
            color="region",
            labels={"region": "Region", "profit": "Profit"}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Profit vs Diskon")
        fig2 = px.scatter(
            filtered_df.sample(min(1000, len(filtered_df))),
            x="discount", y="profit",
            color="category", opacity=0.6,
            labels={"discount": "Diskon", "profit": "Profit"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Form prediksi risiko
    st.markdown("---")
    st.subheader("🔮 Prediksi Risiko Transaksi")

    if models.get("risk") is not None:
        with st.form("risk_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                r_qty = st.number_input("Quantity", min_value=1, value=5)
                r_price = st.number_input("Unit Price", min_value=0.0, value=100000.0)
                r_discount = st.slider("Diskon (%)", 0, 80, 20) / 100
            with col_b:
                r_sales = st.number_input("Sales", min_value=0.0, value=500000.0)
                r_profit = st.number_input("Profit (bisa negatif)", value=-10000.0)

            r_submitted = st.form_submit_button("Cek Risiko ▶")

        if r_submitted:
            try:
                r_input = np.array([[r_qty, r_price, r_discount, r_sales, r_profit]])
                r_pred = models["risk"].predict(r_input)[0]
                r_prob = models["risk"].predict_proba(r_input)[0]

                risk_level = ["Rendah 🟢", "Sedang 🟡", "Tinggi 🔴"]
                label_map = {0: "Rendah 🟢", 1: "Sedang 🟡", 2: "Tinggi 🔴"}

                st.metric("Level Risiko", label_map.get(r_pred, str(r_pred)))
                st.bar_chart(pd.DataFrame({"Probabilitas": r_prob}, index=["Rendah", "Sedang", "Tinggi"]))
            except Exception as e:
                st.error(f"Prediksi gagal: {e}")
    else:
        st.info("Model risk tidak ditemukan. Pastikan path model sudah benar.")

# ─────────────────────────────────────────────
# HALAMAN 4: RECOMMENDATION
# ─────────────────────────────────────────────
elif page == "🎯 Recommendation":
    st.title("🎯 Product Recommendation")

    st.subheader("Rekomendasi Berdasarkan Kategori & Region")

    col1, col2 = st.columns(2)
    with col1:
        rec_region = st.selectbox(
            "Pilih Region",
            df["region"].unique().tolist()
        )
    with col2:
        rec_category = st.selectbox(
            "Pilih Kategori",
            df["category"].unique().tolist()
        )

    if st.button("Tampilkan Rekomendasi ▶"):
        mask = (df["region"] == rec_region) & (df["category"] == rec_category)
        top_rec = (
            df[mask]
            .groupby("product_name")
            .agg(
                total_sales=("sales", "sum"),
                total_qty=("quantity", "sum"),
                avg_profit=("profit", "mean"),
                trending=("is_trending", "mean") if "is_trending" in df.columns else ("sales", "count"),
            )
            .sort_values("total_sales", ascending=False)
            .head(10)
            .reset_index()
        )

        if len(top_rec) > 0:
            fig = px.bar(
                top_rec, x="total_sales", y="product_name",
                orientation="h", color="avg_profit",
                color_continuous_scale="RdYlGn",
                labels={
                    "total_sales": "Total Penjualan",
                    "product_name": "Produk",
                    "avg_profit": "Rata-rata Profit"
                },
                title=f"Top 10 Produk — {rec_category} di {rec_region}"
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_rec, use_container_width=True)
        else:
            st.warning("Tidak ada data untuk kombinasi filter ini.")

# ─────────────────────────────────────────────
# HALAMAN 5: REGION ANALYSIS
# ─────────────────────────────────────────────
elif page == "🗺️ Region Analysis":
    st.title("🗺️ Region Analysis")

    # Ringkasan per region
    region_summary = (
        filtered_df.groupby("region")
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            total_orders=("order_id", "count"),
            avg_discount=("discount", "mean"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Penjualan per Region")
        fig = px.pie(
            region_summary, values="total_sales", names="region",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Profit vs Penjualan per Region")
        fig2 = px.scatter(
            region_summary, x="total_sales", y="total_profit",
            size="total_orders", color="region",
            text="region", size_max=50,
            labels={"total_sales": "Total Penjualan", "total_profit": "Total Profit"}
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: region vs kategori
    st.subheader("Heatmap: Penjualan Region × Kategori")
    pivot = (
        filtered_df.groupby(["region", "category"])["sales"]
        .sum()
        .unstack(fill_value=0)
    )
    fig3 = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Kategori", y="Region", color="Total Penjualan")
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Tabel ringkasan
    st.subheader("Tabel Ringkasan Region")
    region_summary["avg_discount"] = (region_summary["avg_discount"] * 100).round(1).astype(str) + "%"
    region_summary.columns = ["Region", "Total Penjualan", "Total Profit", "Total Order", "Rata-rata Diskon"]
    st.dataframe(region_summary, use_container_width=True)