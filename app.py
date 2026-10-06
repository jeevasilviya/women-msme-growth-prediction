import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import os
import time
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Women MSME Growth Predictor",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("women_msme_growth_model.pkl")


try:
    model = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    model_error = str(e)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "owner_age",
    "education_level",
    "business_experience_years",
    "business_age_years",
    "entrepreneurship_training",
    "digital_literacy",
    "family_support",
    "business_network",
    "initial_investment_inr",
    "finance_access",
    "loan_access",
    "financial_literacy",
    "reinvestment_capacity",
    "employees",
    "previous_annual_revenue_inr",
    "monthly_customers",
    "customer_retention",
    "inventory_management",
    "supplier_access",
    "digital_marketing",
    "social_media_presence",
    "online_sales",
    "digital_payment_adoption",
    "digital_inventory_management",
    "competition_level",
    "market_access",
    "product_diversity",
    "seasonal_demand",
    "government_scheme_awareness",
    "pmegp_awareness",
    "pmegp_assistance",
    "mudra_access",
    "standup_india_awareness",
    "government_training",
    "government_subsidy",
    "government_market_linkage",
    "owner_household_size"
]


# ============================================================
# ACTIONABLE FACTORS
# ============================================================

ACTIONABLE_FACTORS = [
    "education_level",
    "entrepreneurship_training",
    "digital_literacy",
    "family_support",
    "business_network",
    "finance_access",
    "loan_access",
    "financial_literacy",
    "reinvestment_capacity",
    "customer_retention",
    "inventory_management",
    "supplier_access",
    "digital_marketing",
    "social_media_presence",
    "online_sales",
    "digital_payment_adoption",
    "digital_inventory_management",
    "market_access",
    "product_diversity",
    "government_scheme_awareness",
    "pmegp_awareness",
    "pmegp_assistance",
    "mudra_access",
    "standup_india_awareness",
    "government_training",
    "government_subsidy",
    "government_market_linkage"
]


# ============================================================
# SHAP SETUP
# ============================================================

@st.cache_resource
def load_shap_explainer():

    background = pd.read_csv("shap_background.csv")

    preprocessor = model.named_steps["preprocessor"]
    regression_model = model.named_steps["model"]

    background_transformed = preprocessor.transform(
        background
    )

    encoded_feature_names = (
        preprocessor.get_feature_names_out()
    )

    explainer = shap.LinearExplainer(
        regression_model,
        background_transformed
    )

    return (
        explainer,
        preprocessor,
        encoded_feature_names
    )


if model_loaded:

    try:

        (
            explainer,
            shap_preprocessor,
            encoded_feature_names
        ) = load_shap_explainer()

        shap_loaded = True

    except Exception as e:

        shap_loaded = False
        shap_error = str(e)

else:

    shap_loaded = False


# ============================================================
# MAP ENCODED FEATURES BACK TO ORIGINAL FACTORS
# ============================================================

def get_original_factor(encoded_name):

    if encoded_name.startswith("num__"):

        return encoded_name.replace(
            "num__",
            ""
        )

    if encoded_name.startswith("cat__"):

        clean_name = encoded_name.replace(
            "cat__",
            ""
        )

        categorical_features = (
            shap_preprocessor.transformers_[1][2]
        )

        for factor in sorted(
            categorical_features,
            key=len,
            reverse=True
        ):

            if clean_name.startswith(
                factor + "_"
            ):
                return factor

    return encoded_name


if shap_loaded:

    original_factor_map = [
        get_original_factor(name)
        for name in encoded_feature_names
    ]


# ============================================================
# GEMINI AI RECOMMENDATIONS
# ============================================================

@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not available in the environment.")
    return genai.Client(api_key=api_key)


def generate_ai_recommendations(attention_factors, simulation_changes, predicted_growth, potential_growth):
    client = get_gemini_client()

    weak_factors = []
    for _, row in attention_factors.iterrows():
        weak_factors.append({
            "factor": row["Original_Factor"].replace("_", " ").title(),
            "current_level": str(row["Current_Value"]),
            "shap_contribution": round(float(row["SHAP_Value"]), 3),
        })

    readable_changes = []
    for change in simulation_changes:
        readable_changes.append({
            "factor": change["Factor"].replace("_", " ").title(),
            "current_value": str(change["Current Value"]),
            "simulated_value": str(change["Simulated Value"]),
        })

    prompt = f"""
You are an AI business-support assistant for a research prototype about women-owned MSMEs.
Generate personalized, practical recommendations using ONLY the supplied model context.

Current model-predicted annual growth: {predicted_growth:.2f}%
Potential model-predicted annual growth under the controlled what-if scenario: {potential_growth:.2f}%

SHAP-identified actionable factors requiring attention:
{weak_factors}

Controlled simulated improvements:
{readable_changes}

Requirements:
- Give one concise recommendation for each listed weak actionable factor.
- Connect each recommendation to the entrepreneur's current factor level.
- Prefer practical, low-risk, realistic steps suitable for a small business.
- Do not calculate, modify, or invent growth percentages.
- Do not claim that SHAP proves causation.
- Do not promise that a recommendation will increase growth.
- Do not invent eligibility for loans, subsidies, or government schemes.
- If a government-support factor appears, advise the entrepreneur to verify current official eligibility and terms.
- End with one short sentence stating that the recommendations are advisory and model-informed, not guaranteed outcomes.
- Use clear headings or numbered recommendations.
"""

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )
            return response.text

        except Exception as e:
            error_message = str(e)

            # Retry only for temporary Gemini service overloads.
            if "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue

            # Authentication, quota, configuration, and other errors
            # should be surfaced instead of repeatedly retried.
            raise

    raise RuntimeError(
        "Gemini is temporarily busy after multiple attempts."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "📈 Women MSME Growth Prediction System"
)

st.write(
    "AI-based growth prediction and personalized "
    "business recommendation system for women-owned MSMEs."
)

st.divider()


# ============================================================
# SYSTEM STATUS
# ============================================================

if model_loaded:

    st.success(
        "Machine Learning Model Loaded Successfully ✅"
    )

else:

    st.error(
        "Unable to load the Machine Learning model."
    )

    st.error(model_error)

    st.stop()


# ============================================================
# MAIN TABS
# ============================================================

research_tab, predictor_tab = st.tabs(
    [
        "📊 Research Comparison",
        "🤖 AI Growth Predictor"
    ]
)


# ============================================================
# TAB 1 - RESEARCH COMPARISON
# ============================================================

with research_tab:

    # --------------------------------------------------------
    # RESEARCH TAB STYLING
    # --------------------------------------------------------
    st.markdown(
        """
        <style>
        .research-hero {
            padding: 1.35rem 1.5rem;
            border: 1px solid rgba(46, 204, 113, 0.28);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(20, 90, 55, 0.22), rgba(25, 40, 55, 0.30));
            margin-bottom: 1rem;
        }
        .research-hero h2 { margin: 0 0 .35rem 0; }
        .research-hero p { margin: 0; opacity: .86; }
        .insight-card {
            padding: 1rem 1.1rem;
            border-radius: 14px;
            border: 1px solid rgba(120, 180, 150, 0.22);
            background: rgba(30, 45, 40, 0.35);
            min-height: 128px;
        }
        .small-muted { opacity: .72; font-size: .92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="research-hero">
            <h2>📊 Past Research vs Present Model</h2>
            <p>
                Compare findings reported in Indian women-entrepreneurship research
                with the factors identified by the current explainable ML model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "The studies below use different statistical methods and outcomes. "
        "Their numerical results are therefore shown in their original meaning; "
        "they are not converted into artificial 'importance percentages'."
    )

    # --------------------------------------------------------
    # VERIFIED PREVIOUS RESEARCH
    # --------------------------------------------------------
    st.subheader("📚 1. Previous Research")

    previous_research = pd.DataFrame([
        {
            "Year": 2013,
            "Study": "Women entrepreneurs and business venture growth",
            "Context / Sample": "India; N = 158",
            "Factors supported": "Industry experience; prior entrepreneurial experience; business-network size; family support",
            "Reported evidence": "These factors were significant contributors to business growth; education was not significant."
        },
        {
            "Year": 2021,
            "Study": "Digital Marketing Strategies Adopted by Women Entrepreneurs",
            "Context / Sample": "Karnataka; 138 women entrepreneurs",
            "Factors supported": "Digital marketing; digital media / social-media use",
            "Reported evidence": "Digital-marketing adoption was positively associated with sales; Somers' d = 0.645, p < .001 (sales dependent)."
        },
        {
            "Year": 2025,
            "Study": "Financial Access and Entrepreneurship by Gender",
            "Context / Sample": "Rural India; village-level economic-census analysis",
            "Factors supported": "Financial access; formal / institutional credit",
            "Reported evidence": "Bank access within 5 km increased non-agricultural entrepreneurship; the women's effect was driven by institutional-credit uptake."
        },
        {
            "Year": 2025,
            "Study": "An integrated approach of factors influencing entrepreneurial success",
            "Context / Sample": "Odisha; 529 women entrepreneurs",
            "Factors supported": "Social capital; training; government policy; finance; infrastructure; market linkage",
            "Reported evidence": "Structural-equation modelling reported these factors as significant contributors to entrepreneurial success."
        },
        {
            "Year": 2025,
            "Study": "Will She Scale: Determinants that Influence Women-led Enterprise Growth in India",
            "Context / Sample": "5 Indian states; 309 women entrepreneurs",
            "Factors supported": "Initial investment; family support; women-entrepreneur networks; digital platforms; risk-taking; firm resources",
            "Reported evidence": "178 non-scale-up and 131 scale-up enterprises; scale-up used >10% annual sales growth over the previous 3 years."
        },
    ])

    st.dataframe(
        previous_research,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            "Study": st.column_config.TextColumn(width="large"),
            "Factors supported": st.column_config.TextColumn(width="large"),
            "Reported evidence": st.column_config.TextColumn(width="large"),
        },
    )

    st.caption(
        "Previous-research evidence is displayed using the statistic or conclusion "
        "reported by the study itself. 'Significant' does not mean a SHAP percentage."
    )

    # Selected-literature overlap count (not an importance percentage)
    past_factor_counts = pd.DataFrame({
        "Factor": [
            "Finance Access",
            "Digital Marketing / Platforms",
            "Family Support",
            "Business Network / Social Capital",
            "Business Experience",
            "Entrepreneurship Training",
            "Market Access / Linkage",
            "Initial Investment",
            "Government Policy / Support",
        ],
        "Studies": [2, 2, 2, 3, 1, 1, 1, 1, 1],
    }).sort_values("Studies", ascending=True)

    st.markdown("#### 🔁 Factors recurring in the selected past studies")
    st.caption(
        "The chart counts how many of the five selected studies explicitly identified "
        "the factor or a closely corresponding construct. It is a literature-overlap count, "
        "not an effect size or importance score."
    )
    st.bar_chart(
        past_factor_counts.set_index("Factor"),
        horizontal=True,
        height=410,
        x_label="Number of selected studies",
        y_label="Factor",
    )

    st.divider()

    # --------------------------------------------------------
    # CURRENT MODEL RESULTS
    # --------------------------------------------------------
    st.subheader("🧠 2. Present Research — Current ML / SHAP Results")

    current_shap = pd.DataFrame({
        "Factor": [
            "Finance Access",
            "Digital Marketing",
            "Digital Literacy",
            "Business Experience",
            "Customer Retention",
            "Family Support",
            "Financial Literacy",
            "Social Media Presence",
            "Competition Level",
            "Business Age",
            "Inventory Management",
            "Education Level",
            "Supplier Access",
            "Online Sales",
            "Reinvestment Capacity",
        ],
        "Relative SHAP Contribution (%)": [
            6.739, 5.789, 5.314, 5.251, 5.158,
            4.429, 4.299, 4.163, 3.996, 3.839,
            3.690, 3.636, 3.470, 3.057, 2.975,
        ],
    })

    top1, top2, top3 = st.columns(3)
    with top1:
        st.metric("Highest Current Factor", "Finance Access", "6.739% relative SHAP")
    with top2:
        st.metric("Second", "Digital Marketing", "5.789% relative SHAP")
    with top3:
        st.metric("Third", "Digital Literacy", "5.314% relative SHAP")

    st.markdown("#### 📈 Current model factor importance")
    st.bar_chart(
        current_shap.sort_values("Relative SHAP Contribution (%)", ascending=True).set_index("Factor"),
        horizontal=True,
        height=560,
        x_label="Relative mean |SHAP| contribution (%)",
        y_label="Factor",
    )

    st.warning(
        "These percentages are normalized relative mean absolute SHAP contributions. "
        "They describe how much each factor contributes to the model's explanations on average; "
        "they are not causal business-growth percentages."
    )

    st.divider()

    # --------------------------------------------------------
    # PAST VS PRESENT COMPARISON
    # --------------------------------------------------------
    st.subheader("🔄 3. Past vs Present Comparison")

    comparison = pd.DataFrame([
        ["Finance Access", "Supported: finance / formal credit", "6.739%", "Agreement"],
        ["Digital Marketing", "Supported: digital marketing / platforms", "5.789%", "Agreement"],
        ["Digital Literacy", "Digital skills discussed in digital-adoption research", "5.314%", "Strong present-model relevance"],
        ["Business Experience", "Significant contributor in 2013 Indian study", "5.251%", "Agreement"],
        ["Customer Retention", "Less emphasized in these five selected studies", "5.158%", "Notable present-model factor"],
        ["Family Support", "Supported in 2013 and 2025 scale-up research", "4.429%", "Agreement"],
        ["Financial Literacy", "Related to the broader finance-capability theme", "4.299%", "Present-model emphasis"],
        ["Social Media Presence", "Related to digital-media adoption in 2021 study", "4.163%", "Agreement / overlap"],
    ], columns=["Factor", "Past Research Evidence", "Present Relative SHAP", "Interpretation"])

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Factor": st.column_config.TextColumn(width="medium"),
            "Past Research Evidence": st.column_config.TextColumn(width="large"),
            "Present Relative SHAP": st.column_config.TextColumn(width="small"),
            "Interpretation": st.column_config.TextColumn(width="medium"),
        },
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="insight-card">
            <b>💰 Finance</b><br><br>
            Past Indian evidence identifies financial access / formal credit as relevant.
            In the present model, <b>Finance Access</b> has the highest relative SHAP contribution.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="insight-card">
            <b>📱 Digital capability</b><br><br>
            Past research reports a positive digital-marketing–sales association.
            The present model also places Digital Marketing and Digital Literacy near the top.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="insight-card">
            <b>🤝 Human & social capital</b><br><br>
            Business experience, networks and family support appear in prior Indian research;
            experience and family support are also influential in the present model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # --------------------------------------------------------
    # INTERPRETATION + SOURCES
    # --------------------------------------------------------
    st.subheader("💡 4. Research Interpretation")

    st.success(
        "The present model broadly agrees with past Indian women-entrepreneurship research "
        "on the relevance of finance access, digital adoption, business experience and "
        "family/social support. The current model additionally gives notable relative "
        "importance to customer retention, digital literacy and financial literacy."
    )

    st.markdown(
        """
        **How to interpret this comparison:** Past studies establish evidence using methods such as
        regression, correlation and structural-equation modelling. The present study uses an ML
        regression model and SHAP to quantify *relative model contribution*. Therefore, the comparison
        is about **agreement or differences in identified factors**, not a direct comparison of unlike
        numerical statistics.

        **Important limitation:** The current ML model was trained on a synthetic Coimbatore women-MSME
        dataset. Its SHAP results describe the behaviour of this research prototype and should be
        validated with real longitudinal MSME data before making population-level or causal claims.
        """
    )

    with st.expander("📑 View research sources"):
        st.markdown(
            """
            1. **Prasad et al. (2013)** — *Women entrepreneurs and business venture growth: an examination of the influence of human and social capital resources in an Indian context.*  
               https://doi.org/10.1080/08276331.2013.821758

            2. **Digital Marketing Strategies Adopted by Women Entrepreneurs and its Impact on Business Performance (2021)** — Karnataka micro and small enterprises.  
               https://indianjournals.com/article/ijmie-11-4-007

            3. **Garg, Gupta & Mallick (2025 issue; published online 2024)** — *Financial Access and Entrepreneurship by Gender: Evidence from Rural India.*  
               https://doi.org/10.1007/s11187-024-00925-z

            4. **Mahato & Jha (2025)** — *An integrated approach of factors influencing entrepreneurial success: evidence from women-owned MSMEs in India.*  
               https://doi.org/10.1504/IJESB.2025.145868

            5. **Prabha et al. (2025)** — *Will She Scale: Determinants that Influence Women-led Enterprise Growth in India.*  
               https://doi.org/10.1016/j.wsif.2025.103138
            """
        )


# ============================================================
# TAB 2 - AI GROWTH PREDICTOR
# ============================================================

with predictor_tab:

    st.header(
        "🤖 Entrepreneur Growth Analysis"
    )

    st.write(
        "Enter the entrepreneur and business "
        "information below. The trained "
        "machine-learning model will estimate "
        "annual business growth."
    )


    # ========================================================
    # INPUT FORM
    # ========================================================

    with st.form(
        "entrepreneur_form"
    ):


        # ====================================================
        # 1. ENTREPRENEUR PROFILE
        # ====================================================

        st.subheader(
            "1️⃣ Entrepreneur Profile"
        )

        col1, col2 = st.columns(2)


        with col1:

            owner_age = st.number_input(
                "Owner Age",
                min_value=18,
                max_value=80,
                value=35
            )

            education_level = st.selectbox(
                "Education Level",
                [
                    "School",
                    "Diploma",
                    "Undergraduate",
                    "Postgraduate"
                ]
            )

            business_experience_years = (
                st.number_input(
                    "Business Experience (Years)",
                    min_value=0.0,
                    max_value=60.0,
                    value=5.0,
                    step=0.1
                )
            )

            business_age_years = (
                st.number_input(
                    "Business Age (Years)",
                    min_value=0.0,
                    max_value=60.0,
                    value=3.0,
                    step=0.1
                )
            )

            entrepreneurship_training = (
                st.selectbox(
                    "Entrepreneurship Training",
                    ["No", "Yes"]
                )
            )


        with col2:

            digital_literacy = st.selectbox(
                "Digital Literacy",
                ["Low", "Medium", "High"]
            )

            family_support = st.selectbox(
                "Family Support",
                ["Low", "Medium", "High"]
            )

            business_network = st.selectbox(
                "Business Network",
                ["Low", "Medium", "High"]
            )

            owner_household_size = (
                st.number_input(
                    "Household Size",
                    min_value=1,
                    max_value=20,
                    value=4
                )
            )


        st.divider()


        # ====================================================
        # 2. FINANCIAL FACTORS
        # ====================================================

        st.subheader(
            "2️⃣ Financial Factors"
        )

        col1, col2 = st.columns(2)


        with col1:

            initial_investment_inr = (
                st.number_input(
                    "Initial Investment (₹)",
                    min_value=0.0,
                    value=100000.0,
                    step=10000.0
                )
            )

            finance_access = st.selectbox(
                "Finance Access",
                [
                    "Limited",
                    "Moderate",
                    "Good"
                ]
            )

            loan_access = st.selectbox(
                "Loan Access",
                ["No", "Yes"]
            )


        with col2:

            financial_literacy = (
                st.selectbox(
                    "Financial Literacy",
                    [
                        "Low",
                        "Medium",
                        "High"
                    ]
                )
            )

            reinvestment_capacity = (
                st.selectbox(
                    "Reinvestment Capacity",
                    [
                        "Low",
                        "Medium",
                        "High"
                    ]
                )
            )

            previous_annual_revenue_inr = (
                st.number_input(
                    "Previous Annual Revenue (₹)",
                    min_value=0.0,
                    value=500000.0,
                    step=50000.0
                )
            )


        st.divider()


        # ====================================================
        # 3. BUSINESS OPERATIONS
        # ====================================================

        st.subheader(
            "3️⃣ Business Operations"
        )

        col1, col2 = st.columns(2)


        with col1:

            employees = st.number_input(
                "Number of Employees",
                min_value=0,
                max_value=500,
                value=5
            )

            monthly_customers = st.number_input(
                "Monthly Customers",
                min_value=0,
                max_value=100000,
                value=100
            )

            customer_retention = (
                st.selectbox(
                    "Customer Retention",
                    [
                        "Low",
                        "Medium",
                        "High"
                    ]
                )
            )


        with col2:

            inventory_management = (
                st.selectbox(
                    "Inventory Management",
                    [
                        "Manual",
                        "Basic_Digital",
                        "Advanced_Digital"
                    ]
                )
            )

            supplier_access = st.selectbox(
                "Supplier Access",
                [
                    "Difficult",
                    "Moderate",
                    "Good"
                ]
            )


        st.divider()


        # ====================================================
        # 4. DIGITAL ADOPTION
        # ====================================================

        st.subheader(
            "4️⃣ Digital Adoption"
        )

        col1, col2 = st.columns(2)


        with col1:

            digital_marketing = st.selectbox(
                "Digital Marketing",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )

            social_media_presence = (
                st.selectbox(
                    "Social Media Presence",
                    [
                        "No",
                        "Basic",
                        "Active"
                    ]
                )
            )

            online_sales = st.selectbox(
                "Online Sales",
                ["No", "Yes"]
            )


        with col2:

            digital_payment_adoption = (
                st.selectbox(
                    "Digital Payment Adoption",
                    [
                        "Low",
                        "Medium",
                        "High"
                    ]
                )
            )

            digital_inventory_management = (
                st.selectbox(
                    "Digital Inventory Management",
                    [
                        "No",
                        "Basic",
                        "Advanced"
                    ]
                )
            )


        st.divider()


        # ====================================================
        # 5. MARKET FACTORS
        # ====================================================

        st.subheader(
            "5️⃣ Market Factors"
        )

        col1, col2 = st.columns(2)


        with col1:

            competition_level = (
                st.selectbox(
                    "Competition Level",
                    [
                        "Low",
                        "Medium",
                        "High"
                    ]
                )
            )

            market_access = st.selectbox(
                "Market Access",
                [
                    "Limited",
                    "Moderate",
                    "Good"
                ]
            )


        with col2:

            product_diversity = (
                st.selectbox(
                    "Product Diversity",
                    [
                        "Low",
                        "Medium",
                        "High"
                    ]
                )
            )

            seasonal_demand = (
                st.selectbox(
                    "Seasonal Demand",
                    [
                        "Low",
                        "Medium",
                        "High"
                    ]
                )
            )


        st.divider()


        # ====================================================
        # 6. GOVERNMENT SUPPORT
        # ====================================================

        st.subheader(
            "6️⃣ Government Support"
        )

        col1, col2 = st.columns(2)


        with col1:

            government_scheme_awareness = (
                st.selectbox(
                    "Government Scheme Awareness",
                    [
                        "Low",
                        "Medium",
                        "High"
                    ]
                )
            )

            pmegp_awareness = st.selectbox(
                "PMEGP Awareness",
                ["No", "Yes"]
            )

            pmegp_assistance = st.selectbox(
                "PMEGP Assistance",
                ["No", "Yes"]
            )

            mudra_access = st.selectbox(
                "MUDRA Access",
                ["No", "Yes"]
            )


        with col2:

            standup_india_awareness = (
                st.selectbox(
                    "Stand-Up India Awareness",
                    ["No", "Yes"]
                )
            )

            government_training = (
                st.selectbox(
                    "Government Training",
                    ["No", "Yes"]
                )
            )

            government_subsidy = (
                st.selectbox(
                    "Government Subsidy",
                    ["No", "Yes"]
                )
            )

            government_market_linkage = (
                st.selectbox(
                    "Government Market Linkage",
                    ["No", "Yes"]
                )
            )


        st.divider()


        # ====================================================
        # SUBMIT
        # ====================================================

        analyze_button = (
            st.form_submit_button(
                "🔍 Analyze My Business",
                use_container_width=True
            )
        )


    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze_button:


        # ====================================================
        # CREATE INPUT DICTIONARY
        # ====================================================

        user_input = {

            "owner_age":
                owner_age,

            "education_level":
                education_level,

            "business_experience_years":
                business_experience_years,

            "business_age_years":
                business_age_years,

            "entrepreneurship_training":
                entrepreneurship_training,

            "digital_literacy":
                digital_literacy,

            "family_support":
                family_support,

            "business_network":
                business_network,

            "initial_investment_inr":
                initial_investment_inr,

            "finance_access":
                finance_access,

            "loan_access":
                loan_access,

            "financial_literacy":
                financial_literacy,

            "reinvestment_capacity":
                reinvestment_capacity,

            "employees":
                employees,

            "previous_annual_revenue_inr":
                previous_annual_revenue_inr,

            "monthly_customers":
                monthly_customers,

            "customer_retention":
                customer_retention,

            "inventory_management":
                inventory_management,

            "supplier_access":
                supplier_access,

            "digital_marketing":
                digital_marketing,

            "social_media_presence":
                social_media_presence,

            "online_sales":
                online_sales,

            "digital_payment_adoption":
                digital_payment_adoption,

            "digital_inventory_management":
                digital_inventory_management,

            "competition_level":
                competition_level,

            "market_access":
                market_access,

            "product_diversity":
                product_diversity,

            "seasonal_demand":
                seasonal_demand,

            "government_scheme_awareness":
                government_scheme_awareness,

            "pmegp_awareness":
                pmegp_awareness,

            "pmegp_assistance":
                pmegp_assistance,

            "mudra_access":
                mudra_access,

            "standup_india_awareness":
                standup_india_awareness,

            "government_training":
                government_training,

            "government_subsidy":
                government_subsidy,

            "government_market_linkage":
                government_market_linkage,

            "owner_household_size":
                owner_household_size
        }


        user_df = pd.DataFrame(
            [user_input]
        )

        user_df = user_df[FEATURES]


        # ====================================================
        # CURRENT GROWTH PREDICTION
        # ====================================================

        try:

            predicted_growth = (
                model.predict(user_df)[0]
            )


            st.divider()

            st.header(
                "📈 Growth Prediction Result"
            )


            metric_col1, metric_col2, metric_col3 = (
                st.columns([1, 2, 1])
            )


            with metric_col2:

                st.metric(
                    "Predicted Annual Growth",
                    f"{predicted_growth:.2f}%"
                )


            st.info(
                "This percentage is a "
                "machine-learning model prediction "
                "based on the supplied business "
                "factors. It is not a guaranteed "
                "future growth rate."
            )


            # ================================================
            # SHAP ANALYSIS
            # ================================================

            st.subheader(
                "🔎 Explainable AI Analysis"
            )


            if not shap_loaded:

                st.error(
                    "SHAP explainer could not be loaded."
                )

                st.error(shap_error)

            else:

                user_transformed = (
                    shap_preprocessor.transform(
                        user_df
                    )
                )

                user_shap = explainer(
                    user_transformed
                )

                encoded_values = (
                    user_shap.values[0]
                )


                shap_df = pd.DataFrame({

                    "Original_Factor":
                        original_factor_map,

                    "SHAP_Value":
                        encoded_values
                })


                # --------------------------------------------
                # AGGREGATE ENCODED FEATURES
                # --------------------------------------------

                shap_37 = (
                    shap_df
                    .groupby(
                        "Original_Factor",
                        as_index=False
                    )["SHAP_Value"]
                    .sum()
                )


                shap_37[
                    "Current_Value"
                ] = (
                    shap_37[
                        "Original_Factor"
                    ]
                    .map(user_input)
                )


                shap_37[
                    "Impact_Strength"
                ] = (
                    shap_37[
                        "SHAP_Value"
                    ].abs()
                )


                # --------------------------------------------
                # SUPPORTING FACTORS
                # --------------------------------------------

                supporting_factors = (
                    shap_37[
                        shap_37[
                            "SHAP_Value"
                        ] > 0
                    ]
                    .sort_values(
                        "SHAP_Value",
                        ascending=False
                    )
                    .head(5)
                )


                # --------------------------------------------
                # ACTIONABLE NEGATIVE FACTORS
                # --------------------------------------------

                attention_factors = (
                    shap_37[
                        (
                            shap_37[
                                "SHAP_Value"
                            ] < 0
                        )
                        &
                        (
                            shap_37[
                                "Original_Factor"
                            ].isin(
                                ACTIONABLE_FACTORS
                            )
                        )
                    ]
                    .sort_values(
                        "SHAP_Value"
                    )
                    .head(5)
                    .reset_index(
                        drop=True
                    )
                )


                # --------------------------------------------
                # DISPLAY SUPPORTING FACTORS
                # --------------------------------------------

                st.write(
                    "### ✅ Factors Supporting "
                    "the Prediction"
                )


                if len(
                    supporting_factors
                ) > 0:

                    support_display = (
                        supporting_factors[
                            [
                                "Original_Factor",
                                "Current_Value",
                                "SHAP_Value"
                            ]
                        ].copy()
                    )


                    support_display[
                        "Original_Factor"
                    ] = (
                        support_display[
                            "Original_Factor"
                        ]
                        .str.replace(
                            "_",
                            " "
                        )
                        .str.title()
                    )


                    support_display.columns = [
                        "Factor",
                        "Current Level",
                        "SHAP Contribution"
                    ]


                    support_display[
                        "SHAP Contribution"
                    ] = (
                        support_display[
                            "SHAP Contribution"
                        ].round(3)
                    )


                    st.dataframe(
                        support_display,
                        use_container_width=True,
                        hide_index=True
                    )


                # --------------------------------------------
                # DISPLAY ATTENTION FACTORS
                # --------------------------------------------

                st.write(
                    "### ⚠️ Factors Requiring "
                    "Attention"
                )


                if len(
                    attention_factors
                ) > 0:

                    attention_display = (
                        attention_factors[
                            [
                                "Original_Factor",
                                "Current_Value",
                                "SHAP_Value"
                            ]
                        ].copy()
                    )


                    attention_display[
                        "Original_Factor"
                    ] = (
                        attention_display[
                            "Original_Factor"
                        ]
                        .str.replace(
                            "_",
                            " "
                        )
                        .str.title()
                    )


                    attention_display.columns = [
                        "Factor",
                        "Current Level",
                        "SHAP Contribution"
                    ]


                    attention_display[
                        "SHAP Contribution"
                    ] = (
                        attention_display[
                            "SHAP Contribution"
                        ].round(3)
                    )


                    st.dataframe(
                        attention_display,
                        use_container_width=True,
                        hide_index=True
                    )


                    st.caption(
                        "Negative SHAP values indicate "
                        "factors that lowered this model "
                        "prediction relative to its SHAP "
                        "baseline. They do not establish "
                        "causal effects."
                    )


                else:

                    st.success(
                        "No actionable negative factors "
                        "were identified for this "
                        "prediction."
                    )


                # ============================================
                # POTENTIAL GROWTH SIMULATION
                # ============================================

                st.subheader("🚀 Potential Growth Simulation")

                improvement_maps = {
                    "education_level": {"School": "Diploma", "Diploma": "Undergraduate", "Undergraduate": "Postgraduate", "Postgraduate": "Postgraduate"},
                    "entrepreneurship_training": {"No": "Yes", "Yes": "Yes"},
                    "digital_literacy": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "family_support": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "business_network": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "finance_access": {"Limited": "Moderate", "Moderate": "Good", "Good": "Good"},
                    "loan_access": {"No": "Yes", "Yes": "Yes"},
                    "financial_literacy": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "reinvestment_capacity": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "customer_retention": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "inventory_management": {"Manual": "Basic_Digital", "Basic_Digital": "Advanced_Digital", "Advanced_Digital": "Advanced_Digital"},
                    "supplier_access": {"Difficult": "Moderate", "Moderate": "Good", "Good": "Good"},
                    "digital_marketing": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "social_media_presence": {"No": "Basic", "Basic": "Active", "Active": "Active"},
                    "online_sales": {"No": "Yes", "Yes": "Yes"},
                    "digital_payment_adoption": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "digital_inventory_management": {"No": "Basic", "Basic": "Advanced", "Advanced": "Advanced"},
                    "market_access": {"Limited": "Moderate", "Moderate": "Good", "Good": "Good"},
                    "product_diversity": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "government_scheme_awareness": {"Low": "Medium", "Medium": "High", "High": "High"},
                    "pmegp_awareness": {"No": "Yes", "Yes": "Yes"},
                    "pmegp_assistance": {"No": "Yes", "Yes": "Yes"},
                    "mudra_access": {"No": "Yes", "Yes": "Yes"},
                    "standup_india_awareness": {"No": "Yes", "Yes": "Yes"},
                    "government_training": {"No": "Yes", "Yes": "Yes"},
                    "government_subsidy": {"No": "Yes", "Yes": "Yes"},
                    "government_market_linkage": {"No": "Yes", "Yes": "Yes"}
                }

                simulated_df = user_df.copy()
                simulation_changes = []

                for _, factor_row in attention_factors.iterrows():
                    factor = factor_row["Original_Factor"]
                    current_value = simulated_df.iloc[0][factor]

                    if factor in improvement_maps:
                        new_value = improvement_maps[factor].get(current_value, current_value)

                        if new_value != current_value:
                            simulated_df.loc[simulated_df.index[0], factor] = new_value
                            simulation_changes.append({
                                "Factor": factor,
                                "Current Value": current_value,
                                "Simulated Value": new_value
                            })

                if simulation_changes:
                    potential_growth = model.predict(simulated_df)[0]
                    estimated_change = potential_growth - predicted_growth

                    st.write("### Simulated Improvements")
                    changes_display = pd.DataFrame(simulation_changes)
                    changes_display["Factor"] = (
                        changes_display["Factor"].str.replace("_", " ").str.title()
                    )
                    st.dataframe(changes_display, use_container_width=True, hide_index=True)

                    st.write("### Growth Scenario Comparison")
                    growth_col1, growth_col2, growth_col3 = st.columns(3)

                    with growth_col1:
                        st.metric("Current Predicted Growth", f"{predicted_growth:.2f}%")
                    with growth_col2:
                        st.metric("Potential Predicted Growth", f"{potential_growth:.2f}%")
                    with growth_col3:
                        st.metric("Estimated Change", f"{estimated_change:+.2f} pp")

                    if estimated_change > 0:
                        st.success(
                            "Under this simulated improvement scenario, "
                            f"the model estimates a {estimated_change:.2f} "
                            "percentage-point increase in predicted annual growth."
                        )
                    elif estimated_change < 0:
                        st.warning(
                            "The simulated scenario produced a lower model prediction. "
                            "The application reports the model result as-is."
                        )
                    else:
                        st.info("The simulated scenario produced no change in predicted growth.")

                    st.caption(
                        "Potential growth is a model-based what-if estimate using controlled "
                        "one-level changes to SHAP-identified actionable factors. It is not "
                        "a guaranteed or causal future growth rate."
                    )

                                      # ============================================
                    # AI PERSONALIZED RECOMMENDATIONS
                    # ============================================

                    st.subheader("🤖 AI Personalized Recommendations")

                    try:
                        recommendations = generate_ai_recommendations(
                            attention_factors=attention_factors,
                            predicted_growth=predicted_growth,
                            potential_growth=potential_growth,
                            simulation_changes=simulation_changes
                        )

                        st.markdown(recommendations)

                        st.caption(
                            "These recommendations are AI-generated and model-informed. "
                            "They are advisory and do not guarantee business growth."
                        )

                    except Exception as e:
                        st.warning(
                            "AI recommendations are temporarily unavailable. "
                            "The machine-learning prediction and simulation above are still valid."
                        )

                        st.error(f"Gemini error: {e}")

                else:
                    st.info(
                        "No applicable one-level improvements were available for the "
                        "actionable factors identified in this analysis."
                    )


        except Exception as e:

            st.error(
                "An error occurred while analyzing "
                "the entrepreneur."
            )

            st.exception(e)