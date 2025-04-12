import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time  # Added for AI analysis simulation
from fpdf import FPDF
import base64
import io
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.switch_page_button import switch_page
import os
from dotenv import load_dotenv
import json

# Import utility modules
from gold_utils import (
    get_gold_price, get_dxy, get_economic_calendar,
    get_claude_analysis, get_timeframe_analysis, initialize_sample_data,
    create_pdf, get_pdf_download_link
)
from db_utils import (
    init_db, save_event, update_event, delete_event,
    get_events, save_score, get_scores, save_claude_analysis, get_latest_claude_analysis
)

# Load environment variables from .env file
load_dotenv()

# Setup Claude API
try:
    import anthropic

    anthropic_available = True

    # Initialize Anthropic client if API key is available
    if st.secrets["api_key"]["anthropic"]:
        claude_client = anthropic.Anthropic(
            api_key=st.secrets["api_key"]["anthropic"]
        )
        claude_model = st.secrets["api_key"]["anthropic_model"]
        claude_available = True
    else:
        claude_client = None
        claude_model = None
        claude_available = False
except ImportError:
    claude_client = None
    claude_model = None
    anthropic_available = False
    claude_available = False

# Set page config
st.set_page_config(page_title="Gold Fundamental Analysis Dashboard",
                   layout="wide")

# Initialize database
conn = init_db()

# Check if database connection was successful
if conn is None:
    st.error("Failed to connect to the database. Some features may not work properly.")
    # Initialize an empty placeholder for high_impact_events to prevent errors
    high_impact_events = []
else:
    # Get events and initialize high_impact_events
    events = get_events(conn)
    # Initialize high_impact_events with events that have impact level 3
    high_impact_events = [e for e in events if e[4] == 3]  # Assuming impact is at index 4
    
    # Add this line here to initialize sample data if needed
    initialize_sample_data(conn)

# App title
st.title("Gold Fundamental Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to",
                      ["Dashboard", "AI Analysis", "Calendar & Checklist", "Scoring System",
                       "Data Analysis", "Usage Guide", "Settings"])



# Dashboard page
if page == "Dashboard":
    st.header("Economic Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Market Data")

        # Gold price chart
        gold_data = get_gold_price()
        if not gold_data.empty:
            # Make sure the 'Close' column exists
            if 'Close' in gold_data.columns:
                fig = px.line(gold_data, y='Close',
                              title='Gold Price (Last 30 Days)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Find the actual column names
                st.write(f"Available columns: {gold_data.columns.tolist()}")
                # Try to plot using the first column that might be the closing price
                # Often it's just named differently
                if len(gold_data.columns) > 0:
                    # Use the last column if multiple exist (often the price column)
                    price_col = gold_data.columns[-1]
                    fig = px.line(gold_data, y=price_col,
                                  title=f'Gold Price (Last 30 Days) - Using column: {price_col}')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(
                        "Could not find appropriate columns to plot gold price data")

        # DXY chart
        dxy_data = get_dxy()
        if not dxy_data.empty:
            # Make sure the 'Close' column exists
            if 'Close' in dxy_data.columns:
                fig = px.line(dxy_data, y='Close',
                              title='US Dollar Index (Last 30 Days)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Find the actual column names
                st.write(f"Available DXY columns: {dxy_data.columns.tolist()}")
                # Try to plot using the first column that might be the closing price
                if len(dxy_data.columns) > 0:
                    # Use the last column if multiple exist (often the price column)
                    price_col = dxy_data.columns[-1]
                    fig = px.line(dxy_data, y=price_col,
                                  title=f'US Dollar Index (Last 30 Days) - Using column: {price_col}')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(
                        "Could not find appropriate columns to plot DXY data")

    with col2:
        st.subheader("Upcoming Events")

        # Get events for the next 7 days
        events = get_events(conn)
        today = datetime.now().strftime("%Y-%m-%d")
        next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

        upcoming_events = [e for e in events if today <= e[1] <= next_week]

        if upcoming_events:
            event_df = pd.DataFrame(upcoming_events,
                                    columns=['id', 'date', 'title', 'category',
                                             'impact', 'notes', 'completed'])
            event_df['date'] = pd.to_datetime(event_df['date'])
            event_df = event_df.sort_values('date')

            for _, event in event_df.iterrows():
                impact_emoji = "🔴" if event['impact'] == 3 else "🟠" if event[
                                                                           'impact'] == 2 else "🟡"
                st.write(
                    f"{event['date'].strftime('%Y-%m-%d')} - {impact_emoji} **{event['title']}** ({event['category']})")
        else:
            st.write("No upcoming events in the next 7 days.")

        st.subheader("Latest Fundamental Scores")
        scores = get_scores(conn)

        if scores:
            latest_score = scores[0]
            score_id, date, interest_rates, inflation, dollar_strength, supply, demand, positioning, notes = latest_score

            # Format date
            formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime(
                "%B %d, %Y")
            st.write(f"Score from: {formatted_date}")

            # Create radar chart
            categories = ['Interest Rates', 'Inflation', 'Dollar Strength',
                          'Supply', 'Demand', 'Positioning']
            values = [interest_rates, inflation, dollar_strength, supply,
                      demand, positioning]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Fundamental Score'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Overall sentiment calculation
            overall_score = sum(values) / len(values)
            sentiment = "Bullish" if overall_score >= 6.5 else "Neutral" if overall_score >= 3.5 else "Bearish"
            st.metric("Overall Sentiment", sentiment,
                      f"{overall_score:.1f}/10")
        else:
            st.write(
                "No scores available. Add your first score in the Scoring System page.")


# AI Analysis page
elif page == "AI Analysis":
    st.header("Claude AI Market Analysis")

    with st.expander("Gold Market Outlook", expanded=True):
        col1, col2 = st.columns([0.7, 0.3])
        
        # Add a refresh button at the top
        refresh = st.button("Refresh Analysis")
        
        # Get saved analysis or generate new one if refresh is clicked
        analysis_type = "market_outlook"
        saved_analysis, created_at = get_latest_claude_analysis(conn, analysis_type)
        
        if refresh or saved_analysis is None:
            # Get Claude's independent analysis (fresh analysis)
            claude_analysis = get_claude_analysis(conn, claude_client, claude_model, claude_available)
            
            # Save the analysis to the database
            save_claude_analysis(conn, analysis_type, claude_analysis)
        else:
            # Use the saved analysis
            claude_analysis = saved_analysis
            st.info(f"Analysis last updated: {created_at}")

        with col1:
            # Display the analysis
            outlook = claude_analysis["outlook"]
            outlook_color = "#3D9970" if outlook == "Bullish" else "#FF4136" if outlook == "Bearish" else "#FFDC00"

            st.markdown(f"""
            <h3 style="color:{outlook_color}">AI Outlook: {outlook}</h3>
            """, unsafe_allow_html=True)

            if "confidence" in claude_analysis:
                confidence = claude_analysis["confidence"]
                st.progress(confidence / 100, f"Confidence: {confidence}%")

            st.write("**Analysis:**")
            st.write(claude_analysis["analysis"])

            if claude_analysis["key_factors"]:
                st.write("**Key Drivers:**")
                for factor in claude_analysis["key_factors"]:
                    st.write(f"• {factor}")

        with col2:
            if "factor_scores" in claude_analysis:
                # If using the Claude API, we'll use placeholder factor scores for visualization
                # Create a radar chart of the factors (these won't affect the analysis)
                categories = list(claude_analysis["factor_scores"].keys())
                values = list(claude_analysis["factor_scores"].values())

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Market Factors'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    showlegend=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

            # Add a note about the independent analysis
            st.info(
                "This analysis represents Claude's independent market assessment based on current conditions, not your fundamental scores.")

    # Add a timeframe selection for more specific analysis
    st.subheader("Time-Based Analysis")
    timeframe = st.radio("Select Timeframe for Analysis",
                        ["Weekly Outlook", "Monthly Projection",
                        "Quarterly Forecast"], horizontal=True)

    # Refresh button for timeframe analysis
    refresh_timeframe = st.button("Generate Analysis", key="refresh_timeframe")

    # Check if we have a saved analysis for this timeframe
    analysis_type = f"timeframe_{timeframe.lower().replace(' ', '_')}"
    saved_timeframe, timeframe_created_at = get_latest_claude_analysis(conn, analysis_type)

    if refresh_timeframe or saved_timeframe is None:
        with st.spinner(f"Analyzing {timeframe.lower()} trends..."):
            # Get Claude's independent timeframe analysis
            timeframe_analysis = get_timeframe_analysis(conn, timeframe, claude_client, claude_model, claude_available)
            
            # Save as JSON data
            timeframe_data = {"analysis": timeframe_analysis}
            save_claude_analysis(conn, analysis_type, timeframe_data)
            
            # Display the analysis
            st.markdown(timeframe_analysis)
            
            if not refresh_timeframe:
                st.success("Analysis generated successfully! Next time, it will be loaded from cache unless you click 'Generate Analysis'.")
    else:
        # Show when the analysis was created
        st.info(f"Analysis last updated: {timeframe_created_at}")
        
        # Display the saved analysis
        st.markdown(saved_timeframe["analysis"])
        
    # After displaying the timeframe analysis
    if saved_timeframe is not None or refresh_timeframe:
        # Generate PDF
        try:
            pdf_title = f"Gold Market {timeframe} - Analysis Report"
            analysis_content = saved_timeframe["analysis"] if not refresh_timeframe else timeframe_analysis
            pdf_bytes = create_pdf(pdf_title, analysis_content)
            
            # Create download link
            st.markdown(
                get_pdf_download_link(pdf_bytes, f"gold_market_{timeframe.lower().replace(' ', '_')}.pdf"), 
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")


# Calendar & Checklist page
elif page == "Calendar & Checklist":
    st.header("Calendar & Checklist")

    # Store which tab is active in session state
    if 'events_tab' not in st.session_state:
        st.session_state.events_tab = "View Events"

    tab1, tab2 = st.tabs(["View Events", "Add/Edit Events"])

    # Set active tab based on session state
    if st.session_state.events_tab == "Add/Edit Events":
        tab2.write("")  # This activates the tab

    with tab1:
        # Check for event view popup
        if 'view_event' in st.session_state:
            event = st.session_state.view_event
            with st.expander(f"Event Details: {event['title']}",
                             expanded=True):
                st.write(f"**Date:** {event['date'].strftime('%Y-%m-%d')}")
                st.write(f"**Category:** {event['category']}")
                impact_levels = {1: "Low", 2: "Medium", 3: "High"}
                st.write(
                    f"**Impact:** {impact_levels.get(event['impact'], 'Unknown')}")
                st.write(
                    f"**Completed:** {'Yes' if event['completed'] == 1 else 'No'}")
                st.write("**Notes:**")
                st.write(
                    event['notes'] if event['notes'] else "No notes available")

                if st.button("Close", key="close_event_view"):
                    del st.session_state.view_event
                    st.rerun()

        # Get all events
        events = get_events(conn)

        if not events:
            # If no events in DB, populate with sample data
            calendar_events = get_economic_calendar()
            for event in calendar_events:
                save_event(
                    event['date'],
                    event['title'],
                    event['category'],
                    event['impact'],
                    event['notes'],
                    event['completed']
                )
            events = get_events(conn)

        # Convert to DataFrame
        if events:
            df = pd.DataFrame(events,
                              columns=['id', 'date', 'title', 'category',
                                       'impact', 'notes', 'completed'])
            df['date'] = pd.to_datetime(df['date'])

            # Filter options
            st.subheader("Filter Events")
            col1, col2, col3 = st.columns(3)

            with col1:
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()
                start_date = st.date_input("Start Date", min_date)
                end_date = st.date_input("End Date", max_date)

            with col2:
                categories = sorted(df['category'].unique())
                selected_categories = st.multiselect("Categories", categories,
                                                     default=categories)

            with col3:
                impact_options = [1, 2, 3]
                impact_labels = {1: "Low", 2: "Medium", 3: "High"}
                selected_impacts = st.multiselect(
                    "Impact Level",
                    impact_options,
                    default=impact_options,
                    format_func=lambda x: impact_labels[x]
                )

            # Apply filters
            mask = (df['date'].dt.date >= start_date) & (
                        df['date'].dt.date <= end_date)
            if selected_categories:
                mask &= df['category'].isin(selected_categories)
            if selected_impacts:
                mask &= df['impact'].isin(selected_impacts)

            filtered_df = df[mask].sort_values('date')

            if not filtered_df.empty:
                st.subheader("Event Checklist")

                # Group by date
                grouped = filtered_df.groupby(filtered_df['date'].dt.date)

                for date, group in grouped:
                    st.write(f"### {date.strftime('%A, %B %d, %Y')}")

                    for _, row in group.iterrows():
                        # Create a unique key for each checkbox
                        key = f"check_{row['id']}"
                        impact_emoji = "🔴" if row['impact'] == 3 else "🟠" if \
                        row['impact'] == 2 else "🟡"

                        col1, col2, col3 = st.columns([0.05, 0.75, 0.2])

                        with col1:
                            checked = st.checkbox("", value=(
                                        row['completed'] == 1), key=key)
                            if checked != (row['completed'] == 1):
                                # Update completion status
                                update_event(
                                conn,             # Database connection
                                row['id'],        # Event ID
                                row['date'].strftime('%Y-%m-%d'),  # Date
                                row['title'],     # Title
                                row['category'],  # Category
                                row['impact'],    # Impact
                                row['notes'],     # Notes
                                1 if checked else 0  # Completed status
                            )

                        with col2:
                            title_style = "text-decoration: line-through;" if \
                            row['completed'] == 1 else ""
                            st.markdown(
                                f"<span style='{title_style}'>{impact_emoji} **{row['title']}** ({row['category']})</span>",
                                unsafe_allow_html=True)

                        with col3:
                            button_cols = st.columns(2)
                            # View button (opens a popup with details)
                            with button_cols[0]:
                                if st.button("👁️", key=f"view_{row['id']}"):
                                    # Store this event in session state for viewing
                                    st.session_state.view_event = row.to_dict()

                            # Edit button (navigates to edit tab)
                            with button_cols[1]:
                                if st.button("✏️", key=f"edit_{row['id']}"):
                                    # Store this event in session state for editing
                                    st.session_state.edit_event = row.to_dict()
                                    # Switch to edit tab
                                    st.session_state.events_tab = "Add/Edit Events"
                                    st.rerun()

                # Calendar view
                st.subheader("Calendar View")
                calendar_df = filtered_df.copy()
                calendar_df['day'] = calendar_df['date'].dt.day
                calendar_df['month'] = calendar_df['date'].dt.month
                calendar_df['year'] = calendar_df['date'].dt.year

                month_counts = calendar_df.groupby(
                    ['year', 'month', 'day']).size().reset_index(name='count')

                # Create the calendar heatmap
                if not month_counts.empty:
                    fig = px.density_heatmap(
                        month_counts,
                        x='day',
                        y='month',
                        z='count',
                        labels={'day': 'Day', 'month': 'Month',
                                'count': 'Events'},
                        title='Event Calendar Heatmap'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No events match your filter criteria.")
        else:
            st.write(
                "No events available. Add your first event using the form below.")

    with tab2:
        st.subheader("Add/Edit Event")

        # Check if we're editing an event
        if 'edit_event' in st.session_state:
            event = st.session_state.edit_event
            st.write(f"Editing: {event['title']}")
            is_edit = True
        else:
            event = {
                'id': None,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'title': '',
                'category': 'Interest Rates',
                'impact': 2,
                'notes': '',
                'completed': 0
            }
            is_edit = False

        # Event form
        date = st.date_input("Date",
                             value=datetime.strptime(event['date'],
                                                     '%Y-%m-%d') if isinstance(
                                 event['date'], str) else event['date'])
        title = st.text_input("Event Title", value=event['title'])

        col1, col2 = st.columns(2)

        with col1:
            category = st.selectbox(
                "Category",
                ["Interest Rates", "Inflation", "Currency", "Supply", "Demand",
                 "Positioning", "Other"],
                index=["Interest Rates", "Inflation", "Currency", "Supply",
                       "Demand", "Positioning", "Other"].index(
                    event['category']) if event['category'] in [
                    "Interest Rates", "Inflation", "Currency", "Supply",
                    "Demand", "Positioning", "Other"] else 0
            )

        with col2:
            impact = st.select_slider(
                "Impact Level",
                options=[1, 2, 3],
                value=event['impact'],
                format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x]
            )

        notes = st.text_area("Notes", value=event['notes'])
        completed = st.checkbox("Completed", value=event['completed'] == 1)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save Event"):
                if is_edit:
                    update_event(
                        conn,
                        event['id'],
                        date.strftime('%Y-%m-%d'),
                        title,
                        category,
                        impact,
                        notes,
                        1 if completed else 0
                    )
                    st.success(f"Event '{title}' updated successfully!")
                    # Clear edit state
                    if 'edit_event' in st.session_state:
                        del st.session_state.edit_event
                else:
                    save_event(
                        conn,
                        date.strftime('%Y-%m-%d'),
                        title,
                        category,
                        impact,
                        notes,
                        1 if completed else 0
                    )
                    st.success(f"Event '{title}' added successfully!")
                # Using st.rerun() instead of experimental_rerun
                st.rerun()

        with col2:
            if is_edit and st.button("Delete Event", type="primary"):
                delete_event(event['id'])
                st.success(f"Event '{title}' deleted successfully!")
                if 'edit_event' in st.session_state:
                    del st.session_state.edit_event
                st.rerun()

            if is_edit and st.button("Cancel Editing"):
                if 'edit_event' in st.session_state:
                    del st.session_state.edit_event
                st.rerun()

# Scoring System page
elif page == "Scoring System":
    st.header("Fundamental Scoring System")

    tab1, tab2 = st.tabs(["Add New Score", "View Score History"])

    with tab1:
        st.write("""
        Rate each fundamental factor on a scale of 0-10:
        - 0-3: Bearish for gold
        - 4-6: Neutral
        - 7-10: Bullish for gold
        """)

        score_date = st.date_input("Date", value=datetime.now())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Macroeconomic Factors")
            interest_rates = st.slider(
                "Interest Rates & Monetary Policy",
                0, 10, 5,
                help="Higher = lower rates/dovish policy (bullish for gold)"
            )

            inflation = st.slider(
                "Inflation",
                0, 10, 5,
                help="Higher = rising inflation/inflation concerns (bullish for gold)"
            )

            dollar_strength = st.slider(
                "Dollar Strength",
                0, 10, 5,
                help="Higher = weaker dollar (bullish for gold)"
            )

        with col2:
            st.subheader("Gold Specific Factors")
            supply = st.slider(
                "Supply Factors",
                0, 10, 5,
                help="Higher = lower supply/supply constraints (bullish for gold)"
            )

            demand = st.slider(
                "Demand Components",
                0, 10, 5,
                help="Higher = stronger demand (bullish for gold)"
            )

            positioning = st.slider(
                "Market Positioning",
                0, 10, 5,
                help="Higher = favorable positioning (bullish for gold)"
            )

        notes = st.text_area("Analysis Notes")

        if st.button("Save Score Analysis"):
            save_score(
                conn,
                score_date.strftime('%Y-%m-%d'),
                interest_rates,
                inflation,
                dollar_strength,
                supply,
                demand,
                positioning,
                notes
            )
            st.success(
                f"Score analysis saved for {score_date.strftime('%Y-%m-%d')}!")

            # Calculate overall sentiment
            factors = [interest_rates, inflation, dollar_strength, supply,
                       demand, positioning]
            overall_score = sum(factors) / len(factors)
            sentiment = "Bullish" if overall_score >= 6.5 else "Neutral" if overall_score >= 3.5 else "Bearish"

            st.write(f"Overall Score: {overall_score:.1f}/10")
            st.write(f"Sentiment: {sentiment}")

            # Create radar chart
            categories = ['Interest Rates', 'Inflation', 'Dollar Strength',
                          'Supply', 'Demand', 'Positioning']
            values = [interest_rates, inflation, dollar_strength, supply,
                      demand, positioning]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Fundamental Score'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Score History")

        scores = get_scores(conn)

        if scores:
            score_df = pd.DataFrame(scores,
                                    columns=['id', 'date', 'interest_rates',
                                             'inflation', 'dollar_strength',
                                             'supply', 'demand', 'positioning',
                                             'notes'])
            score_df['date'] = pd.to_datetime(score_df['date'])

            # Calculate overall score and sentiment
            score_df['overall_score'] = score_df[
                ['interest_rates', 'inflation', 'dollar_strength',
                 'supply', 'demand', 'positioning']].mean(axis=1)

            score_df['sentiment'] = score_df['overall_score'].apply(
                lambda
                    x: "Bullish" if x >= 6.5 else "Neutral" if x >= 3.5 else "Bearish"
            )

            # Score trend chart
            st.subheader("Score Trends")

            # Melt the dataframe for line chart
            plot_df = score_df.melt(
                id_vars=['date'],
                value_vars=['interest_rates', 'inflation', 'dollar_strength',
                            'supply', 'demand', 'positioning',
                            'overall_score'],
                var_name='Factor',
                value_name='Score'
            )

            # Clean factor names for display
            plot_df['Factor'] = plot_df['Factor'].apply(
                lambda x: ' '.join(x.split('_')).title())
            plot_df['Factor'] = plot_df['Factor'].replace('Overall Score',
                                                          'Overall Score')

            fig = px.line(
                plot_df,
                x='date',
                y='Score',
                color='Factor',
                title='Fundamental Factor Scores Over Time',
                labels={'date': 'Date', 'Score': 'Score (0-10)'}
            )

            fig.update_layout(yaxis_range=[0, 10])
            st.plotly_chart(fig, use_container_width=True)

            # Show the score table
            st.subheader("Score Details")
            display_df = score_df[
                ['date', 'overall_score', 'sentiment', 'notes']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['overall_score'] = display_df['overall_score'].round(1)
            display_df.columns = ['Date', 'Overall Score', 'Sentiment',
                                  'Notes']

            # Format with colored sentiment
            st.dataframe(display_df)
        else:
            st.write(
                "No score history available. Add your first score using the form.")

# Data Analysis page
elif page == "Data Analysis":
    st.header("Gold Market Data Analysis")

    st.info(
        "This section would typically connect to market data APIs to pull and analyze real-time data. For demonstration purposes, we're using sample visualizations.")

    tab1, tab2, tab3 = st.tabs(
        ["Price Analysis", "Correlation Analysis", "Seasonality"])

    with tab1:
        st.subheader("Gold Price Analysis")

        # Sample gold price data
        date_range = pd.date_range(start='2023-01-01', end='2024-04-01',
                                   freq='D')
        np.random.seed(42)  # For reproducibility

        # Simulate gold price with trend and some volatility
        base_price = 1800
        trend = np.linspace(0, 200, len(date_range))
        volatility = np.random.normal(0, 30, len(date_range)).cumsum()
        seasonality = 20 * np.sin(np.linspace(0, 6 * np.pi, len(date_range)))

        gold_price = base_price + trend + volatility + seasonality

        gold_df = pd.DataFrame({
            'Date': date_range,
            'Price': gold_price
        })

        # Calculate moving averages
        gold_df['MA50'] = gold_df['Price'].rolling(window=50).mean()
        gold_df['MA200'] = gold_df['Price'].rolling(window=200).mean()

        # Create price chart with moving averages
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=gold_df['Date'],
            y=gold_df['Price'],
            mode='lines',
            name='Gold Price',
            line=dict(color='gold', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=gold_df['Date'],
            y=gold_df['MA50'],
            mode='lines',
            name='50-Day MA',
            line=dict(color='blue', width=1.5)
        ))

        fig.add_trace(go.Scatter(
            x=gold_df['Date'],
            y=gold_df['MA200'],
            mode='lines',
            name='200-Day MA',
            line=dict(color='red', width=1.5)
        ))

        fig.update_layout(
            title='Gold Price with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Technical indicators
        st.subheader("Technical Indicators")
        col1, col2 = st.columns(2)

        with col1:
            # RSI calculation (simplified)
            delta = gold_df['Price'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            gold_df['RSI'] = 100 - (100 / (1 + rs))

            # RSI Chart
            fig_rsi = go.Figure()

            fig_rsi.add_trace(go.Scatter(
                x=gold_df['Date'],
                y=gold_df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=1.5)
            ))

            # Add overbought/oversold lines
            fig_rsi.add_shape(
                type="line",
                x0=gold_df['Date'].min(),
                y0=70,
                x1=gold_df['Date'].max(),
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
            )

            fig_rsi.add_shape(
                type="line",
                x0=gold_df['Date'].min(),
                y0=30,
                x1=gold_df['Date'].max(),
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
            )

            fig_rsi.update_layout(
                title='Relative Strength Index (RSI)',
                xaxis_title='Date',
                yaxis_title='RSI Value',
                yaxis=dict(range=[0, 100])
            )

            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            # MACD calculation (simplified)
            gold_df['EMA12'] = gold_df['Price'].ewm(span=12,
                                                    adjust=False).mean()
            gold_df['EMA26'] = gold_df['Price'].ewm(span=26,
                                                    adjust=False).mean()
            gold_df['MACD'] = gold_df['EMA12'] - gold_df['EMA26']
            gold_df['Signal'] = gold_df['MACD'].ewm(span=9,
                                                    adjust=False).mean()
            gold_df['Histogram'] = gold_df['MACD'] - gold_df['Signal']

            # MACD Chart
            fig_macd = go.Figure()

            fig_macd.add_trace(go.Scatter(
                x=gold_df['Date'],
                y=gold_df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1.5)
            ))

            fig_macd.add_trace(go.Scatter(
                x=gold_df['Date'],
                y=gold_df['Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=1.5)
            ))

            # Add histogram
            fig_macd.add_trace(go.Bar(
                x=gold_df['Date'],
                y=gold_df['Histogram'],
                name='Histogram',
                marker=dict(
                    color=gold_df['Histogram'].apply(
                        lambda x: 'green' if x > 0 else 'red')
                )
            ))

            fig_macd.update_layout(
                title='Moving Average Convergence Divergence (MACD)',
                xaxis_title='Date',
                yaxis_title='Value'
            )

            st.plotly_chart(fig_macd, use_container_width=True)

    with tab2:
        st.subheader("Correlation Analysis")

        # Create correlation data
        # Sample data for DXY, Interest Rates, Inflation
        np.random.seed(42)

        # Create a date range
        corr_dates = pd.date_range(start='2023-01-01', end='2024-04-01',
                                   freq='D')

        # Create the correlation dataframe
        corr_df = pd.DataFrame({
            'Date': corr_dates,
            'Gold': gold_df['Price'].values,
            'DXY': 101 - 0.01 * gold_df['Price'] + np.random.normal(0, 2,
                                                                    len(corr_dates)),
            'Interest_Rate': 4.5 - 0.001 * gold_df['Price'] + np.random.normal(
                0, 0.2, len(corr_dates)),
            'Inflation': 3 + 0.0005 * gold_df['Price'] + np.random.normal(0,
                                                                          0.3,
                                                                          len(corr_dates)),
            'SPX': 4000 + 0.2 * gold_df['Price'] + np.random.normal(0, 50,
                                                                    len(corr_dates))
        })

        # Calculate rolling correlations
        window = 30  # 30-day rolling correlation

        corr_df['Gold_DXY_Corr'] = corr_df['Gold'].rolling(window).corr(
            corr_df['DXY'])
        corr_df['Gold_Rate_Corr'] = corr_df['Gold'].rolling(window).corr(
            corr_df['Interest_Rate'])
        corr_df['Gold_Inflation_Corr'] = corr_df['Gold'].rolling(window).corr(
            corr_df['Inflation'])
        corr_df['Gold_SPX_Corr'] = corr_df['Gold'].rolling(window).corr(
            corr_df['SPX'])

        # Plot rolling correlations
        corr_plot_df = corr_df.melt(
            id_vars=['Date'],
            value_vars=['Gold_DXY_Corr', 'Gold_Rate_Corr',
                        'Gold_Inflation_Corr', 'Gold_SPX_Corr'],
            var_name='Correlation',
            value_name='Value'
        )

        # Clean correlation names for display
        corr_plot_df['Correlation'] = corr_plot_df['Correlation'].replace({
            'Gold_DXY_Corr': 'Gold vs. US Dollar',
            'Gold_Rate_Corr': 'Gold vs. Interest Rates',
            'Gold_Inflation_Corr': 'Gold vs. Inflation',
            'Gold_SPX_Corr': 'Gold vs. S&P 500'
        })

        fig_corr = px.line(
            corr_plot_df,
            x='Date',
            y='Value',
            color='Correlation',
            title=f'{window}-Day Rolling Correlation with Gold',
            labels={'Date': 'Date', 'Value': 'Correlation Coefficient'}
        )

        fig_corr.add_shape(
            type="line",
            x0=corr_plot_df['Date'].min(),
            y0=0,
            x1=corr_plot_df['Date'].max(),
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
        )

        fig_corr.update_layout(
            yaxis=dict(range=[-1, 1]),
            yaxis_tickvals=[-1, -0.5, 0, 0.5, 1],
            yaxis_ticktext=['-1<br>(Perfect<br>Negative)', '-0.5',
                            '0<br>(No<br>Correlation)', '0.5',
                            '1<br>(Perfect<br>Positive)']
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        # Correlation matrix
        st.subheader("Current Correlation Matrix")

        # Calculate current correlation matrix
        corr_matrix = corr_df[
            ['Gold', 'DXY', 'Interest_Rate', 'Inflation', 'SPX']].corr()

        # Rename columns and index for better display
        corr_matrix.columns = ['Gold', 'US Dollar', 'Interest Rate',
                               'Inflation', 'S&P 500']
        corr_matrix.index = ['Gold', 'US Dollar', 'Interest Rate', 'Inflation',
                             'S&P 500']

        # Create heatmap
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Asset Correlation Matrix'
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.write("""
        **Correlation Interpretation:**
        - **-1.0 to -0.7**: Strong negative correlation
        - **-0.7 to -0.3**: Moderate negative correlation
        - **-0.3 to +0.3**: Weak or no correlation
        - **+0.3 to +0.7**: Moderate positive correlation
        - **+0.7 to +1.0**: Strong positive correlation
        """)

    with tab3:
        st.subheader("Gold Price Seasonality")

        # Create sample seasonality data
        years = list(range(2015, 2024))
        months = list(range(1, 13))

        # Create a dictionary to store monthly returns
        monthly_returns = {}

        # Generate random but somewhat consistent monthly returns
        np.random.seed(123)
        base_returns = np.random.normal(0.5, 2, 12)  # Base monthly returns

        # Generate yearly data with some consistency plus noise
        for year in years:
            yearly_factor = np.random.normal(1, 0.3)  # Yearly variation factor
            yearly_returns = base_returns * yearly_factor + np.random.normal(0,
                                                                             1,
                                                                             12)
            monthly_returns[year] = yearly_returns

        # Convert to DataFrame
        seasonality_data = []
        for year in years:
            for i, month in enumerate(months):
                seasonality_data.append({
                    'Year': year,
                    'Month': month,
                    'Return': monthly_returns[year][i]
                })

        seasonality_df = pd.DataFrame(seasonality_data)

        # Calculate average monthly returns
        avg_monthly_returns = seasonality_df.groupby('Month')[
            'Return'].mean().reset_index()

        # Add month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                       'Sep', 'Oct', 'Nov', 'Dec']
        avg_monthly_returns['Month_Name'] = avg_monthly_returns['Month'].apply(
            lambda x: month_names[x - 1])

        # Create bar chart of average monthly returns
        fig_seasonal = px.bar(
            avg_monthly_returns,
            x='Month_Name',
            y='Return',
            title='Average Monthly Gold Returns (2015-2023)',
            labels={'Month_Name': 'Month', 'Return': 'Average Return (%)'},
            color='Return',
            color_continuous_scale='RdYlGn',
            text_auto='.2f'
        )

        fig_seasonal.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': month_names}
        )

        st.plotly_chart(fig_seasonal, use_container_width=True)

        # Create heatmap of yearly returns by month
        pivot_df = seasonality_df.pivot(index='Year', columns='Month',
                                        values='Return')
        pivot_df.columns = month_names

        fig_heatmap = px.imshow(
            pivot_df,
            text_auto='.1f',
            color_continuous_scale='RdYlGn',
            title='Monthly Gold Returns by Year (%)',
            labels={'x': 'Month', 'y': 'Year', 'color': 'Return (%)'}
        )

        fig_heatmap.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': month_names}
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Best and worst months
        st.subheader("Best and Worst Months for Gold")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Top 3 Best Months:**")
            best_months = avg_monthly_returns.sort_values('Return',
                                                          ascending=False).head(
                3)
            for _, row in best_months.iterrows():
                st.write(f"- **{row['Month_Name']}**: {row['Return']:.2f}%")

        with col2:
            st.write("**Top 3 Worst Months:**")
            worst_months = avg_monthly_returns.sort_values('Return').head(3)
            for _, row in worst_months.iterrows():
                st.write(f"- **{row['Month_Name']}**: {row['Return']:.2f}%")

        # Seasonal patterns explanation
        st.write("""
        **Seasonal Patterns in Gold Trading:**

        Gold often exhibits seasonal patterns that can be attributed to several factors:

        1. **Jewelry Demand**: Typically peaks before major holidays and wedding seasons in India and China
        2. **Investment Demand**: Often increases during times of market uncertainty or at the beginning of the year
        3. **Central Bank Buying**: Sometimes follows particular fiscal year patterns
        4. **Mining Production**: Can vary seasonally due to weather and operational factors

        These patterns should be considered alongside other fundamental and technical factors in your trading strategy.
        """)

# Settings page
elif page == "Settings":
    st.header("Dashboard Settings")

    st.subheader("Data Sources")

    st.write("**Current Data Sources:**")
    data_sources = {
        "Gold Price": "Yahoo Finance (GC=F)",
        "US Dollar Index": "Yahoo Finance (DX=F)",
        "Economic Calendar": "Sample Data (Demo)",
        "ETF Flows": "Not Connected",
        "Central Bank Purchases": "Not Connected"
    }

    for source, value in data_sources.items():
        st.write(f"- **{source}**: {value}")

    st.info(
        "In a real implementation, this section would allow you to connect to various data APIs and customize your data sources.")

    st.subheader("Notification Settings")

    st.write("Set up alerts for important events and threshold conditions:")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Email notifications for high-impact events", value=True)
        st.checkbox("Price alert notifications", value=True)
        st.checkbox("Weekly summary reports", value=True)

    with col2:
        st.checkbox("Correlation change alerts", value=False)
        st.checkbox("Sentiment score alerts", value=False)
        st.checkbox("Mobile push notifications", value=False)

    st.subheader("Export Data")

    st.write("Export your analysis and data for external use:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.button("Export Events to CSV")

    with col2:
        st.button("Export Scores to CSV")

    with col3:
        st.button("Export Charts as PNG")

    st.subheader("About")

    st.write("""
    **Gold Trading Fundamental Analysis Dashboard**

    This dashboard is designed to help gold traders integrate fundamental analysis into their trading strategy.
    It provides tools for tracking key fundamental factors, maintaining an economic calendar, and scoring fundamental conditions.

    **Features:**
    - Economic calendar with event tracking
    - Fundamental scoring system for gold
    - Market data analysis and visualization
    - Correlation and seasonality analysis

    For a real implementation, consider extending this with real API connections, automated data imports, and additional features.
    """)

# Usage Guide page
elif page == "Usage Guide":
    st.header("Usage Guide")

    st.subheader("Quick Tips")

    st.markdown("""
    1. **Start with the Dashboard** for an overview of current market conditions and upcoming events
    2. **Use the Calendar & Checklist** to track important economic releases and gold-specific events
    3. **Record your fundamental analysis** in the Scoring System to track your market sentiment over time
    4. **Dive deeper** with the Data Analysis tools to understand correlations and patterns
    """)

    st.subheader("Detailed Instructions")

    # Dashboard section
    st.markdown("### Dashboard")
    st.markdown("""
    The Dashboard provides a quick overview of:
    - Current gold price and US Dollar Index trends
    - Upcoming events in the next 7 days
    - Your latest fundamental score analysis

    **Best Practice**: Check the dashboard daily to stay updated on market conditions and prepare for upcoming events.
    """)

    # Calendar section
    st.markdown("### Calendar & Checklist")
    st.markdown("""
    The Calendar & Checklist helps you track important events that affect gold prices:

    **Adding Events**:
    1. Navigate to the "Add/Edit Events" tab
    2. Fill in the event details (date, title, category, impact level)
    3. Add optional notes about what to look for
    4. Click "Save Event"

    **Tracking Events**:
    1. Filter events by date range, category, or impact level
    2. Check off events as you analyze them
    3. View the calendar heatmap to see busy periods

    **Categories to Monitor**:
    - Interest Rates: Fed meetings, policy announcements
    - Inflation: CPI, PPI, PCE data
    - Currency: Dollar movements, forex policy changes
    - Supply: Mining output, production disruptions
    - Demand: Central bank purchases, ETF flows, jewelry demand
    - Positioning: COT reports, market sentiment

    **Impact Levels**:
    - 🔴 High: Events that typically move gold prices significantly
    - 🟠 Medium: Events with moderate potential impact
    - 🟡 Low: Minor events for context
    """)

    # Scoring System section
    st.markdown("### Scoring System")
    st.markdown("""
    The Fundamental Scoring System helps quantify your analysis:

    **Scoring Process**:
    1. Rate each fundamental factor on a scale of 0-10
    2. Higher scores (7-10) indicate bullish conditions for gold
    3. Lower scores (0-3) indicate bearish conditions
    4. Middle scores (4-6) indicate neutral conditions

    **Key Factors**:
    - **Interest Rates**: Lower rates/dovish policy is bullish for gold
    - **Inflation**: Higher inflation/inflation expectations is bullish
    - **Dollar Strength**: Weaker dollar is bullish for gold
    - **Supply**: Lower supply/mining constraints is bullish
    - **Demand**: Stronger investment/jewelry demand is bullish
    - **Market Positioning**: Low speculative positioning is bullish (contrarian)

    **Best Practice**: Complete a new score weekly or after major market events to track changing conditions.
    """)

    # Data Analysis section
    st.markdown("### Data Analysis")
    st.markdown("""
    The Data Analysis tools help you understand deeper market patterns:

    **Price Analysis**:
    - Gold price charts with moving averages
    - Technical indicators (RSI, MACD)

    **Correlation Analysis**:
    - How gold relates to other assets over time
    - Current correlation matrix

    **Seasonality**:
    - Historical monthly performance patterns
    - Year-by-year comparison

    **Best Practice**: Use these tools to identify periods when gold may deviate from its fundamental drivers.
    """)

    st.subheader("Trading Strategy Integration")
    st.markdown("""
    This dashboard is designed to complement your technical analysis:

    1. **Long-term Bias**: Use your fundamental score to establish a directional bias
    2. **Entry Timing**: Use technical analysis for specific entry points
    3. **Position Sizing**: Increase size when fundamental and technical align
    4. **Risk Management**: Adjust stop levels based on event volatility expectations

    Remember that gold responds to both short-term technical factors and long-term fundamental drivers. The most successful approach integrates both perspectives.
    """)

# Add a footer
if page != "Usage Guide":  # Only show the mini usage guide if not on the full guide page
    st.markdown("""
    ---
    ### Usage Guide

    **Quick Tips:**
    1. Start with the **Dashboard** for an overview of current market conditions and upcoming events
    2. Use the **Calendar & Checklist** to track important economic releases and gold-specific events
    3. Record your fundamental analysis in the **Scoring System** to track your market sentiment over time
    4. Dive deeper with the **Data Analysis** tools to understand correlations and patterns

    *For more detailed instructions, visit the Usage Guide page in the navigation sidebar.*
    """)
else:
    # Just add a simple footer on the Usage Guide page
    st.markdown("---")
    st.markdown("*Dashboard created for fundamental gold trading analysis*")