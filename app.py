import streamlit as st

st.title("Insurance Wrapper Model")

st.write("This will model:")
st.write("- Taxable accounts")
st.write("- PPVA")
st.write("- PPLI")

investment = st.number_input("Initial Investment", value=10000000)

st.write("Investment:", investment)
