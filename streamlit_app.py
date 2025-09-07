import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pyRothC.RothC import RothC

# Set page config
st.set_page_config(
    page_title="RothC SOC Simulator",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #228B22;
        border-bottom: 2px solid #228B22;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #228B22;
        margin: 1rem 0;
    }
    .weather-table {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🌱 RothC Soil Organic Carbon Simulator</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>About this Simulator:</strong><br>
This tool uses the RothC model to simulate soil organic carbon dynamics over time. 
Configure the soil parameters in the sidebar and weather data in the table below to see how SOC changes.
</div>
""", unsafe_allow_html=True)

# Sidebar for soil parameters
st.sidebar.markdown('<h2 class="section-header">🏔️ Soil Parameters</h2>', unsafe_allow_html=True)

# Soil thickness parameter
soil_thick = st.sidebar.slider(
    "Soil Thickness (cm)", 
    min_value=10, 
    max_value=100, 
    value=25, 
    step=5,
    help="Organic layer topsoil thickness in centimeters"
)

# SOC parameter
SOC = st.sidebar.slider(
    "Initial SOC (Mg/ha)", 
    min_value=10.0, 
    max_value=200.0, 
    value=69.7, 
    step=0.1,
    help="Initial soil organic carbon content in Mg per hectare"
)

# Clay content parameter
clay = st.sidebar.slider(
    "Clay Content (%)", 
    min_value=0, 
    max_value=100, 
    value=48, 
    step=1,
    help="Percentage of clay in the soil"
)

# Input carbon parameter
input_carbon = st.sidebar.slider(
    "Annual Carbon Input (Mg/ha/yr)", 
    min_value=0.0, 
    max_value=10.0, 
    value=2.7, 
    step=0.1,
    help="Annual carbon inputs to soil in Mg per hectare per year"
)

# Simulation years parameter
years = st.sidebar.slider(
    "Simulation Years", 
    min_value=10, 
    max_value=50, 
    value=20, 
    step=5,
    help="Number of years to simulate"
)

# Main content area
st.markdown('<h2 class="section-header">🌤️ Weather Data Configuration</h2>', unsafe_allow_html=True)

# Default weather data
default_temp = [-0.4, 0.3, 4.2, 8.3, 13.0, 15.9, 18.0, 17.5, 13.4, 8.7, 3.9, 0.6]
default_precip = [49, 39, 44, 41, 61, 58, 71, 58, 51, 48, 50, 58]
default_evp = [12, 18, 35, 58, 82, 90, 97, 84, 54, 31, 14, 10]

# Create weather data table
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Initialize session state for weather data if not exists
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = pd.DataFrame({
        'Month': months,
        'Temperature (°C)': default_temp,
        'Precipitation (mm)': default_precip,
        'Evaporation (mm)': default_evp
    })

# Create columns for weather data input
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Monthly Weather Data")
    st.markdown("Edit the values directly in the table below:")
    
    # Create editable dataframe
    edited_weather = st.data_editor(
        st.session_state.weather_data,
        column_config={
            "Month": st.column_config.TextColumn("Month", disabled=True),
            "Temperature (°C)": st.column_config.NumberColumn("Temperature (°C)", min_value=-50, max_value=50, step=0.1),
            "Precipitation (mm)": st.column_config.NumberColumn("Precipitation (mm)", min_value=0, max_value=500, step=1),
            "Evaporation (mm)": st.column_config.NumberColumn("Evaporation (mm)", min_value=0, max_value=200, step=1),
        },
        hide_index=True,
        use_container_width=True
    )

with col2:
    st.markdown("### Quick Actions")
    
    if st.button("🔄 Reset to Default", help="Reset weather data to default values"):
        st.session_state.weather_data = pd.DataFrame({
            'Month': months,
            'Temperature (°C)': default_temp,
            'Precipitation (mm)': default_precip,
            'Evaporation (mm)': default_evp
        })
        st.rerun()
    
    # Quick climate presets
    st.markdown("**Climate Presets:**")
    
    if st.button("🌡️ Warmer Climate (+3°C)"):
        st.session_state.weather_data['Temperature (°C)'] = [t + 3 for t in default_temp]
        st.rerun()
    
    if st.button("❄️ Cooler Climate (-3°C)"):
        st.session_state.weather_data['Temperature (°C)'] = [t - 3 for t in default_temp]
        st.rerun()
    
    if st.button("🌧️ Wetter Climate (+20mm)"):
        st.session_state.weather_data['Precipitation (mm)'] = [p + 20 for p in default_precip]
        st.rerun()

# Update session state with edited data
st.session_state.weather_data = edited_weather

# Simulation section
st.markdown('<h2 class="section-header">🚀 Run Simulation</h2>', unsafe_allow_html=True)

if st.button("🔬 Run RothC Simulation", type="primary", use_container_width=True):
    try:
        with st.spinner("Running RothC simulation..."):
            # Extract weather data arrays
            Temp = np.array(edited_weather['Temperature (°C)'].tolist())
            Precip = np.array(edited_weather['Precipitation (mm)'].tolist())
            Evp = np.array(edited_weather['Evaporation (mm)'].tolist())
            
            # Calculate IOM using Falloon et al. (1998) equation
            IOM = 0.049 * SOC ** (1.139)
            
            # Set numpy print options
            np.set_printoptions(precision=3, suppress=True)
            
            # Initialize RothC model
            rothC = RothC(
                temperature=Temp,
                precip=Precip,
                evaporation=Evp,
                clay=clay,
                years=years,
                input_carbon=input_carbon,
                pE=1.0,
                C0=np.array([0, 0, 0, 0, IOM]),
            )
            
            # Run simulation
            df = rothC.compute()
            df.index = rothC.t
            
            # Display results
            st.markdown('<h2 class="section-header">📊 Simulation Results</h2>', unsafe_allow_html=True)
            
            # Create two columns for results
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                # Create interactive plot with Plotly
                fig = go.Figure()
                
                # Get the carbon pool names
                pool_names = df.columns.tolist()
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
                
                for i, pool in enumerate(pool_names):
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[pool],
                        mode='lines',
                        name=pool,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'<b>{pool}</b><br>Year: %{{x}}<br>C Stock: %{{y:.2f}} Mg/ha<extra></extra>'
                    ))
                
                fig.update_layout(
                    title='Carbon Pool Evolution Over Time',
                    xaxis_title='Years',
                    yaxis_title='C stocks (Mg/ha)',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Total SOC over time
                total_soc = df.sum(axis=1)
                fig_total = px.line(
                    x=df.index, 
                    y=total_soc,
                    title='Total Soil Organic Carbon Over Time',
                    labels={'x': 'Years', 'y': 'Total SOC (Mg/ha)'}
                )
                fig_total.update_traces(line_color='#2E8B57', line_width=3)
                fig_total.update_layout(height=400)
                st.plotly_chart(fig_total, use_container_width=True)
            
            with result_col2:
                st.subheader("📈 Summary Statistics")
                
                # Calculate summary statistics
                initial_total = total_soc.iloc[0]
                final_total = total_soc.iloc[-1]
                max_total = total_soc.max()
                min_total = total_soc.min()
                
                st.metric("Initial Total SOC", f"{initial_total:.2f} Mg/ha")
                st.metric("Final Total SOC", f"{final_total:.2f} Mg/ha", f"{final_total - initial_total:.2f}")
                st.metric("Maximum SOC", f"{max_total:.2f} Mg/ha")
                st.metric("Minimum SOC", f"{min_total:.2f} Mg/ha")
                
                # Final pool distribution
                st.subheader("🧬 Final Pool Distribution")
                final_pools = df.iloc[-1]
                
                for pool, value in final_pools.items():
                    percentage = (value / final_total) * 100
                    st.write(f"**{pool}**: {value:.2f} Mg/ha ({percentage:.1f}%)")
                
                # Model parameters used
                st.subheader("⚙️ Parameters Used")
                st.write(f"**Soil Thickness**: {soil_thick} cm")
                st.write(f"**Clay Content**: {clay}%")
                st.write(f"**Annual C Input**: {input_carbon} Mg/ha/yr")
                st.write(f"**IOM**: {IOM:.2f} Mg/ha")
                st.write(f"**Simulation Years**: {years}")
            
            # Download section
            st.markdown('<h3 class="section-header">💾 Download Results</h3>', unsafe_allow_html=True)
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Prepare CSV data
                csv_data = df.copy()
                csv_data['Total_SOC'] = total_soc
                csv_data.reset_index(inplace=True)
                csv_data.rename(columns={'index': 'Year'}, inplace=True)
                
                csv_string = csv_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_string,
                    file_name=f"rothc_simulation_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_dl2:
                # Prepare parameter summary
                params_summary = {
                    'Parameter': ['Soil Thickness (cm)', 'Initial SOC (Mg/ha)', 'Clay Content (%)', 
                                 'Annual C Input (Mg/ha/yr)', 'IOM (Mg/ha)', 'Simulation Years'],
                    'Value': [soil_thick, SOC, clay, input_carbon, IOM, years]
                }
                params_df = pd.DataFrame(params_summary)
                params_csv = params_df.to_csv(index=False)
                
                st.download_button(
                    label="⚙️ Download Parameters",
                    data=params_csv,
                    file_name="simulation_parameters.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
    except Exception as e:
        st.error(f"Error running simulation: {str(e)}")
        st.info("Please check your input parameters and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
🌱 Built with Streamlit and pyRothC • For research and educational purposes<br>
<em>RothC model developed by Rothamsted Research</em>
</div>
""", unsafe_allow_html=True)
