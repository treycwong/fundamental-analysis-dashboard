# gold_utils.py
from db_utils import save_event, save_score
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yfinance as yf
import streamlit as st


def get_gold_price():
    try:
        # Try downloading gold price data
        gold = yf.download("GC=F", period="1mo")

        # If successful but empty, fall back to sample data
        if gold.empty:
            raise Exception("Empty data returned from yfinance")

        # Check if we have MultiIndex columns and flatten them if needed
        if isinstance(gold.columns, pd.MultiIndex):
            # For MultiIndex columns like [('Close', 'GC=F')], take the first level
            gold = gold.droplevel(1, axis=1)

        return gold
    except Exception as e:
        # Return sample data with consistent column names
        sample_data = pd.DataFrame({
            'Date': pd.date_range(start=datetime.now() - timedelta(days=30),
                                  periods=30),
            'Close': np.random.normal(1900, 20, 30)
        }).set_index('Date')
        return sample_data


def get_dxy():
    try:
        # Try downloading DXY data
        dxy = yf.download("DX=F", period="1mo")

        # If successful but empty, fall back to sample data
        if dxy.empty:
            raise Exception("Empty data returned from yfinance")

        # Check if we have MultiIndex columns and flatten them if needed
        if isinstance(dxy.columns, pd.MultiIndex):
            # For MultiIndex columns like [('Close', 'DX=F')], take the first level
            dxy = dxy.droplevel(1, axis=1)

        return dxy
    except Exception as e:
        # Return sample data with consistent column names
        sample_data = pd.DataFrame({
            'Date': pd.date_range(start=datetime.now() - timedelta(days=30),
                                  periods=30),
            'Close': np.random.normal(104, 1, 30)
        }).set_index('Date')
        return sample_data


def get_economic_calendar():
    # This would normally fetch from an API
    # For demo purposes, we'll create sample data
    today = datetime.now()
    calendar = []

    # Sample economic events
    events = [
        ("Fed Interest Rate Decision", "Interest Rates", 3),
        ("US CPI Data Release", "Inflation", 3),
        ("US PPI Data Release", "Inflation", 2),
        ("ECB Policy Meeting", "Interest Rates", 2),
        ("US Dollar Index Technical Review", "Currency", 2),
        ("Gold Production Report", "Supply", 2),
        ("India Gold Import Data", "Demand", 2),
        ("CFTC Commitment of Traders Report", "Positioning", 2),
        ("China Gold Reserve Update", "Demand", 1),
        ("Gold ETF Flows Weekly Update", "Demand", 1)
    ]

    # Generate events for next 30 days
    for i in range(30):
        date = today + timedelta(days=i)
        # Add 1-3 random events for each day with 40% probability
        if np.random.random() < 0.4:
            num_events = np.random.randint(1, 4)
            for _ in range(num_events):
                event = events[np.random.randint(0, len(events))]
                calendar.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "title": event[0],
                    "category": event[1],
                    "impact": event[2],
                    "notes": "",
                    "completed": 0
                })

    return calendar


# Move the Claude analysis functions to this file as well
def get_claude_analysis(conn, claude_client=None, claude_model=None,
                        claude_available=False):
    """
    Get Claude's independent analysis of the gold market without relying on user scores.
    """

    if conn is None:
        st.error("Database connection not available for analysis")
        return {
            "outlook": "Neutral",
            "confidence": 50,
            "analysis": "Unable to access database for analysis.",
            "key_factors": ["Database connection error"],
            "factor_scores": {
                "Interest Rates": 5, "Inflation": 5, "Dollar Strength": 5,
                "Supply": 5, "Demand": 5, "Market Positioning": 5
            }
        }

    # Get current date for context
    today = datetime.now().strftime("%Y-%m-%d")

    # Get upcoming events for context - wrap this in try/except
    try:
        c = conn.cursor()
        next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        c.execute(
            'SELECT * FROM events WHERE date BETWEEN ? AND ? ORDER BY date, impact DESC',
            (today, next_week))
        upcoming_events = c.fetchall()
    except Exception as e:
        st.error(f"Error retrieving events: {str(e)}")
        upcoming_events = []


    # Get gold price data
    gold_data = get_gold_price()

    # Get dollar index data
    dxy_data = get_dxy()

    # Check if Claude API is available
    if claude_available and claude_client:
        try:
            # Create a prompt for Claude that doesn't reference user's fundamental scores
            prompt = f"""
            As a gold market analyst, provide your independent assessment of the current gold market outlook.

            Today's date: {today}

            Recent market data:
            - Gold has been trading around ${int(gold_data['Close'].iloc[-1])} per ounce
            - The US Dollar Index is at {dxy_data['Close'].iloc[-1]:.2f}

            Upcoming key events in the next 7 days:
            """

            # Add events if available
            if upcoming_events:
                for event in upcoming_events[
                             :5]:  # Limit to 5 most important events
                    event_date = event[1]
                    event_title = event[2]
                    event_category = event[3]
                    event_impact = "High" if event[4] == 3 else "Medium" if \
                    event[4] == 2 else "Low"
                    prompt += f"- {event_date}: {event_title} ({event_category}, {event_impact} impact)\n"
            else:
                prompt += "- No major events scheduled\n"

            prompt += """
            Please provide:
            1. Your overall market outlook (Bullish, Neutral, or Bearish) with a confidence percentage (between 50-100%)
            2. A brief analysis explaining your outlook (100-150 words)
            3. The key factors driving your outlook (list 2-3)

            Your analysis should consider global economic factors, geopolitics, monetary policy trends, inflation outlook, and technical market factors.

            Format your response as JSON with the following structure:
            {
                "outlook": "Bullish/Neutral/Bearish",
                "confidence": 75, // percentage between 50-100
                "analysis": "Your analysis text here...",
                "key_factors": ["Factor 1", "Factor 2", "Factor 3"]
            }
            """

            # Call Claude API
            message = claude_client.messages.create(
                model=claude_model,
                max_tokens=1000,
                temperature=0.2,
                system="You are an expert gold market analyst providing concise, insightful analysis based on current market conditions. Format your response as valid JSON only.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse the response
            try:
                # Extract the response content
                response_text = message.content[0].text

                # Clean the response to ensure it's valid JSON
                if "```json" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[
                        0].strip()
                elif "```" in response_text:
                    json_text = response_text.split("```")[1].strip()
                else:
                    json_text = response_text.strip()

                # Parse the JSON
                claude_response = json.loads(json_text)

                # Add default factor scores for radar chart (won't be used for analysis)
                claude_response["factor_scores"] = {
                    "Interest Rates": 5,
                    "Inflation": 5,
                    "Dollar Strength": 5,
                    "Supply": 5,
                    "Demand": 5,
                    "Market Positioning": 5
                }

                return claude_response

            except json.JSONDecodeError as e:
                # Fall back to simplified analysis
                return generate_fallback_analysis(gold_data, dxy_data,
                                                  upcoming_events)

        except Exception as e:
            # Fall back to simplified analysis
            return generate_fallback_analysis(gold_data, dxy_data,
                                              upcoming_events)
    else:
        # Use simplified analysis if Claude is not available
        return generate_fallback_analysis(gold_data, dxy_data, upcoming_events)


def generate_fallback_analysis(gold_data, dxy_data, upcoming_events):
    """
    Generate a simplified market analysis when Claude API is not available.
    This uses basic technical signals instead of fundamental scores.
    """
    # Simple technical analysis based on price trends
    try:
        # Calculate short and long-term moving averages
        gold_data['MA10'] = gold_data['Close'].rolling(window=10).mean()
        gold_data['MA30'] = gold_data['Close'].rolling(window=30).mean()

        # Get latest closing price and moving averages
        latest_price = gold_data['Close'].iloc[-1]
        ma10 = gold_data['MA10'].iloc[-1]
        ma30 = gold_data['MA30'].iloc[-1]

        # Calculate recent price change (5-day)
        price_change_5d = (latest_price / gold_data['Close'].iloc[
            -6] - 1) * 100

        # Check dollar correlation (simplified)
        latest_dxy = dxy_data['Close'].iloc[-1]
        dxy_change_5d = (latest_dxy / dxy_data['Close'].iloc[-6] - 1) * 100

        # Determine outlook based on simple technical signals
        # Bullish: Price above MAs, positive momentum, dollar weakness
        # Bearish: Price below MAs, negative momentum, dollar strength

        signals = 0
        key_factors = []

        # Signal 1: Moving average relationship
        if ma10 > ma30:
            signals += 1
            key_factors.append("Positive MA crossover (10-day above 30-day)")
        elif ma10 < ma30:
            signals -= 1
            key_factors.append("Negative MA crossover (10-day below 30-day)")

        # Signal 2: Price relative to MAs
        if latest_price > ma10 and latest_price > ma30:
            signals += 1
            key_factors.append("Price trading above key moving averages")
        elif latest_price < ma10 and latest_price < ma30:
            signals -= 1
            key_factors.append("Price trading below key moving averages")

        # Signal 3: Recent momentum
        if price_change_5d > 2:
            signals += 1
            key_factors.append(
                f"Strong positive momentum (+{price_change_5d:.1f}% over 5 days)")
        elif price_change_5d < -2:
            signals -= 1
            key_factors.append(
                f"Strong negative momentum ({price_change_5d:.1f}% over 5 days)")

        # Signal 4: Dollar correlation (inverse relationship)
        if dxy_change_5d < -1:
            signals += 1
            key_factors.append(
                f"Dollar weakness ({dxy_change_5d:.1f}% over 5 days)")
        elif dxy_change_5d > 1:
            signals -= 1
            key_factors.append(
                f"Dollar strength (+{dxy_change_5d:.1f}% over 5 days)")

        # Signal 5: Event risk
        high_impact_events = [e for e in upcoming_events if e[4] == 3]
        if len(high_impact_events) >= 3:
            key_factors.append(
                f"Multiple high-impact events upcoming ({len(high_impact_events)} in next week)")

        # Determine overall outlook based on signals
        if signals >= 2:
            outlook = "Bullish"
            confidence = min(100, 60 + signals * 8)
            analysis = f"Gold appears bullish at ${latest_price:.2f} with multiple positive technical signals. "
            analysis += "Price is showing momentum above key moving averages. "
            if dxy_change_5d < 0:
                analysis += "Dollar weakness is providing additional support. "
            analysis += f"The market has {len(high_impact_events)} high-impact events in the coming week that could add volatility."
        elif signals <= -2:
            outlook = "Bearish"
            confidence = min(100, 60 + abs(signals) * 8)
            analysis = f"Gold appears bearish at ${latest_price:.2f} with multiple negative technical signals. "
            analysis += "Price is trading below key moving averages with downward momentum. "
            if dxy_change_5d > 0:
                analysis += "Dollar strength is creating additional headwinds. "
            analysis += f"The market has {len(high_impact_events)} high-impact events in the coming week that could add volatility."
        else:
            outlook = "Neutral"
            confidence = 50 + abs(signals) * 10
            analysis = f"Gold appears to be in a neutral phase at ${latest_price:.2f} with mixed technical signals. "
            analysis += "The price is consolidating without a clear directional bias. "
            analysis += f"The market has {len(high_impact_events)} high-impact events in the coming week that could provide directional catalysts."

        # Keep only the top 3 key factors if more exist
        if len(key_factors) > 3:
            key_factors = key_factors[:3]

        # Create factor scores for radar chart (simplified)
        factor_scores = {
            "Interest Rates": 5,
            "Inflation": 5,
            "Dollar Strength": 3 if dxy_change_5d > 1 else 7 if dxy_change_5d < -1 else 5,
            "Supply": 5,
            "Demand": 5,
            "Market Positioning": 7 if signals > 0 else 3 if signals < 0 else 5
        }

        return {
            "outlook": outlook,
            "confidence": confidence,
            "analysis": analysis,
            "key_factors": key_factors,
            "factor_scores": factor_scores
        }
    except Exception as e:
        # Ultra fallback if even the technical analysis fails
        return {
            "outlook": "Neutral",
            "confidence": 50,
            "analysis": "Unable to generate detailed market analysis. Please check market data sources and try again.",
            "key_factors": ["Data unavailable for analysis"],
            "factor_scores": {
                "Interest Rates": 5, "Inflation": 5, "Dollar Strength": 5,
                "Supply": 5, "Demand": 5, "Market Positioning": 5
            }
        }


def get_timeframe_analysis(conn, timeframe, claude_client=None,
                           claude_model=None, claude_available=False):
    """
    Get Claude's independent analysis for a specific timeframe without relying on user scores.
    """
    # Get current date for context
    today = datetime.now().strftime("%Y-%m-%d")

    # Determine timeframe parameters
    if timeframe == "Weekly Outlook":
        next_period = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        period_name = "week"
    elif timeframe == "Monthly Projection":
        next_period = (datetime.now() + timedelta(days=30)).strftime(
            "%Y-%m-%d")
        period_name = "month"
    else:  # Quarterly
        next_period = (datetime.now() + timedelta(days=90)).strftime(
            "%Y-%m-%d")
        period_name = "quarter"

    # Get event data for the selected timeframe
    c = conn.cursor()
    c.execute(
        'SELECT * FROM events WHERE date BETWEEN ? AND ? ORDER BY date, impact DESC',
        (today, next_period))
    upcoming_events = c.fetchall()

    # Get gold price data
    gold_data = get_gold_price()

    # Get dollar index data
    dxy_data = get_dxy()

    # Check if Claude API is available
    if claude_available and claude_client:
        try:
            # Create a prompt for Claude that doesn't reference user's fundamental scores
            prompt = f"""
            As a gold market analyst, provide your assessment of the gold market outlook for the next {period_name}.

            Today's date: {today}

            Recent market data:
            - Gold has been trading around ${int(gold_data['Close'].iloc[-1])} per ounce
            - The US Dollar Index is at {dxy_data['Close'].iloc[-1]:.2f}

            Upcoming events during this {period_name}:
            """

            # Add events if available
            if upcoming_events:
                for event in upcoming_events[
                             :10]:  # Limit to 10 most important events
                    event_date = event[1]
                    event_title = event[2]
                    event_category = event[3]
                    event_impact = "High" if event[4] == 3 else "Medium" if \
                    event[4] == 2 else "Low"
                    prompt += f"- {event_date}: {event_title} ({event_category}, {event_impact} impact)\n"
            else:
                prompt += "- No major events scheduled\n"

            prompt += f"""
            Please provide:
            1. A detailed {timeframe.lower()} for gold prices
            2. Analysis of how upcoming events might impact gold prices
            3. Specific tactical considerations for traders during this {period_name}

            Your analysis should consider global economic factors, geopolitics, monetary policy trends, inflation outlook, and technical market factors.

            Format your response in markdown with clear sections.
            """

            # Call Claude API
            message = claude_client.messages.create(
                model=claude_model,
                max_tokens=1000,
                temperature=0.3,
                system=f"You are an expert gold market analyst providing a {timeframe.lower()} for gold prices. Be concise but insightful.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Return the raw markdown response
            return message.content[0].text

        except Exception as e:
            # Fall back to simplified analysis
            return generate_fallback_timeframe_analysis(timeframe, period_name,
                                                        upcoming_events,
                                                        gold_data, dxy_data)
    else:
        # Use simplified analysis if Claude is not available
        return generate_fallback_timeframe_analysis(timeframe, period_name,
                                                    upcoming_events, gold_data,
                                                    dxy_data)


def generate_fallback_timeframe_analysis(timeframe, period_name,
                                         upcoming_events, gold_data, dxy_data):
    """
    Generate a simplified timeframe analysis when Claude API is not available.
    """
    analysis = f"## {timeframe}\n\n"

    # Add some basic market context
    latest_price = gold_data['Close'].iloc[-1]
    analysis += f"Gold is currently trading at ${latest_price:.2f} per ounce. "

    # Calculate short-term trends
    price_change_5d = (latest_price / gold_data['Close'].iloc[-6] - 1) * 100
    if price_change_5d > 0:
        analysis += f"The price has risen {price_change_5d:.1f}% over the past 5 trading days. "
    else:
        analysis += f"The price has fallen {abs(price_change_5d):.1f}% over the past 5 trading days. "

    # Add event analysis
    if not upcoming_events:
        analysis += f"\n\n### Event Analysis\n\nNo significant events found for the selected {period_name}. This could mean lower volatility and more technically-driven price action."
    else:
        # Count high impact events
        high_impact_events = [e for e in upcoming_events if e[4] == 3]
        medium_impact_events = [e for e in upcoming_events if e[4] == 2]

        analysis += f"\n\n### Event Analysis\n\n"

        if len(high_impact_events) >= 3:
            analysis += f"The upcoming {period_name} has **{len(high_impact_events)} high-impact events** that could significantly influence gold prices. "
            analysis += "Expect potential for increased volatility during event releases.\n\n"
        elif len(medium_impact_events) >= 5:
            analysis += f"While there are few high-impact events, the {len(medium_impact_events)} medium-impact events could collectively influence the market direction.\n\n"
        else:
            analysis += f"The upcoming {period_name} has relatively few market-moving events, suggesting potentially lower volatility in gold prices.\n\n"

        # Group events by category
        event_categories = {}
        for event in upcoming_events:
            category = event[3]
            if category not in event_categories:
                event_categories[category] = []
            event_categories[category].append(event)

        # Check for event category concentration
        dominant_categories = sorted(event_categories.items(),
                                     key=lambda x: len(x[1]), reverse=True)

        if dominant_categories:
            top_category, top_events = dominant_categories[0]
            if len(top_events) >= 3:
                analysis += f"There's a concentration of events related to **{top_category}** which may be particularly influential this {period_name}.\n\n"

            # List key upcoming events
            analysis += "Key events to watch:\n\n"
            important_events = [e for e in upcoming_events if e[4] >= 2][:5]
            for event in important_events:
                event_date = datetime.strptime(event[1], '%Y-%m-%d').strftime(
                    '%b %d')
                event_title = event[2]
                event_category = event[3]
                event_impact = "🔴" if event[4] == 3 else "🟠" if event[
                                                                    4] == 2 else "🟡"
                analysis += f"- {event_date}: {event_impact} **{event_title}** ({event_category})\n"

    # Add tactical considerations
    analysis += "\n\n### Tactical Considerations\n\n"

    # Add some generic tactical advice based on price action
    if price_change_5d > 2:
        analysis += "- Gold is showing strong momentum; consider strategies that benefit from continuation\n"
        analysis += "- Watch for breakouts above recent resistance levels with increasing volume as confirmation\n"
        analysis += "- Be mindful of overbought conditions that could lead to short-term pullbacks\n"
    elif price_change_5d < -2:
        analysis += "- Gold is showing downward momentum; be cautious with new long positions\n"
        analysis += "- Monitor support levels for potential breakdowns that could accelerate selling pressure\n"
        analysis += "- Look for signs of capitulation or oversold conditions that might signal reversal opportunities\n"
    else:
        analysis += "- Gold is trading in a consolidation pattern; range-bound strategies may be appropriate\n"
        analysis += "- Consider reduced position sizes until a clearer directional bias emerges\n"
        analysis += "- Watch for breakouts from the current range, which could signal the next directional move\n"

    # Event-based tactics
    if high_impact_events:
        next_big_event = min(high_impact_events,
                             key=lambda x: datetime.strptime(x[1], '%Y-%m-%d'))
        event_date = datetime.strptime(next_big_event[1], '%Y-%m-%d').strftime(
            '%B %d')
        analysis += f"- Consider adjusting position sizes ahead of {next_big_event[2]} on {event_date}\n"

    # Add disclaimer
    analysis += "\n\n> **Disclaimer:** This analysis is based on current market conditions and upcoming events. It should be used as one input among many in your trading decisions. Always conduct your own research and risk assessment."

    return analysis



def initialize_sample_data(conn):
    """Populate the database with sample data if it's empty."""
    if conn is None:
        st.error("Database connection not available")
        return
    
    try:
        # Check if events table is empty
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM events")
        event_count = c.fetchone()[0]
        
        if event_count == 0:
            # Add sample events
            sample_events = [
                ('2025-04-15', 'Fed Interest Rate Decision', 'Interest Rates', 3, 'Expected to keep rates unchanged', 0),
                ('2025-04-18', 'US CPI Data Release', 'Inflation', 2, 'Expected to show moderation in inflation', 0),
                ('2025-04-20', 'Gold ETF Flows Weekly Update', 'Demand', 1, 'Monitor for investment demand trends', 0),
                ('2025-04-25', 'US Dollar Index Report', 'Currency', 2, 'Watch for dollar strength/weakness', 0),
                ('2025-05-01', 'Mining Output Data', 'Supply', 2, 'Check production trends', 0),
            ]
            
            for event in sample_events:
                # Make sure to pass conn as the first argument to save_event
                save_event(conn, event[0], event[1], event[2], event[3], event[4], event[5])
                
        # Check if scores table is empty
        c.execute("SELECT COUNT(*) FROM scores")
        score_count = c.fetchone()[0]
        
        if score_count == 0:
            # Add a sample score analysis
            today = datetime.now().strftime('%Y-%m-%d')
            save_score(
                conn,
                today,
                interest_rates=6,
                inflation=7,
                dollar_strength=5,
                supply=6,
                demand=7,
                positioning=5,
                notes="Initial fundamental analysis. Rising inflation and steady demand appear bullish for gold."
            )
            
    except Exception as e:
        st.error(f"Error initializing sample data: {str(e)}")

def create_pdf(title, content):
    """
    Create a PDF report from the given content
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Set up the PDF
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    
    # Date and header
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 5, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)
    
    # Convert markdown content to simple text
    pdf.set_font("Arial", "", 11)
    
    # Split content by lines and add to PDF
    content_lines = content.split('\n')
    for line in content_lines:
        # Basic markdown handling
        if line.startswith('## '):
            pdf.set_font("Arial", "B", 14)
            pdf.ln(5)
            pdf.cell(0, 10, line[3:], ln=True)
            pdf.set_font("Arial", "", 11)
        elif line.startswith('### '):
            pdf.set_font("Arial", "B", 12)
            pdf.ln(3)
            pdf.cell(0, 7, line[4:], ln=True)
            pdf.set_font("Arial", "", 11)
        elif line.startswith('- '):
            pdf.set_x(15)  # Indent bullet points
            pdf.multi_cell(0, 5, "• " + line[2:])
        elif line.strip() == "":
            pdf.ln(3)
        else:
            pdf.multi_cell(0, 5, line)
    
    # Return the PDF as a bytes object
    return pdf.output(dest='S').encode('latin1')

def get_pdf_download_link(pdf_bytes, filename):
    """Generate a download link for the PDF"""
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href