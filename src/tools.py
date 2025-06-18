# src/tools.py

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool
from typing import List, Callable, Optional
import pytz
from datetime import datetime
from src.utils import logger
from functools import wraps
import requests
import json
import re
import pandas as pd
import io
from typing import Dict, Any
import os

def safe_tool(func: Callable) -> Callable:
    """
    Decorator to handle tool errors gracefully with logging and detailed error messages.
    
    Args:
        func (Callable): The function to be wrapped.
    
    Returns:
        Callable: The wrapped function with error handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Tool error in {func.__name__}: {str(e)}")
            return f"Error in {func.__name__}: {str(e)}. Please check input or try again."
    return wrapper

@safe_tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Returns the current time in a specified timezone with validation.
    
    Args:
        timezone (str): The timezone name (e.g., 'UTC', 'America/New_York'). Defaults to 'UTC'.
    
    Returns:
        str: Formatted current time or error message if timezone is invalid.
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return f"Current time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"
    except pytz.UnknownTimeZoneError:
        valid_zones = ", ".join(["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"])
        return f"Unknown timezone '{timezone}'. Valid examples: {valid_zones}"

@safe_tool
def calculate(expression: str) -> str:
    """
    Safely evaluates mathematical expressions with input validation.
    
    Args:
        expression (str): A mathematical expression (e.g., '2+2', '(5*3)/2').
    
    Returns:
        str: Result of the calculation or error message if invalid.
    """
    allowed_chars = set("0123456789+-*/(). ")
    if not expression or not all(c in allowed_chars for c in expression):
        return "Error: Invalid expression. Use numbers and operators (+, -, *, /, (, )) only."
    
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}. Ensure the expression is valid."

@safe_tool
def get_weather(city: str) -> str:
    """
    Retrieves current weather data for a specified city using OpenWeatherMap API.
    
    Args:
        city (str): City name (e.g., 'London', 'New York').
    
    Returns:
        str: Formatted weather information or error message if request fails.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OpenWeatherMap API key not configured. Please set OPENWEATHER_API_KEY in .env."
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data["cod"] != 200:
            return f"Error: Could not retrieve weather for '{city}'. {data.get('message', 'Unknown error')}"
        
        weather = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        
        return (f"Weather in {city}:\n"
                f"- Conditions: {weather}\n"
                f"- Temperature: {temp}°C\n"
                f"- Humidity: {humidity}%\n"
                f"- Wind Speed: {wind_speed} m/s")
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data for '{city}': {str(e)}"

@safe_tool
def convert_currency(amount: str, from_currency: str, to_currency: str) -> str:
    """
    Converts an amount from one currency to another using ExchangeRate-API.
    
    Args:
        amount (str): Amount to convert (e.g., '100').
        from_currency (str): Source currency code (e.g., 'USD').
        to_currency (str): Target currency code (e.g., 'EUR').
    
    Returns:
        str: Converted amount or error message if request fails.
    """
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    if not api_key:
        return "Error: ExchangeRate-API key not configured. Please set EXCHANGERATE_API_KEY in .env."
    
    try:
        amount = float(amount)
        if amount < 0:
            return "Error: Amount must be non-negative."
        
        base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{from_currency.upper()}"
        response = requests.get(base_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data["result"] != "success":
            return f"Error: Could not retrieve rates for {from_currency}. {data.get('error-type', 'Unknown error')}"
        
        rate = data["conversion_rates"].get(to_currency.upper())
        if not rate:
            return f"Error: Invalid target currency '{to_currency}'. Supported currencies: {', '.join(data['conversion_rates'].keys())}"
        
        converted = amount * rate
        return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"
    except ValueError:
        return "Error: Invalid amount. Please provide a valid number."
    except requests.exceptions.RequestException as e:
        return f"Error fetching exchange rates: {str(e)}"

@safe_tool
def analyze_csv(file_content: str) -> str:
    """
    Analyzes a CSV file content and provides basic statistical insights.
    
    Args:
        file_content (str): CSV content as a string.
    
    Returns:
        str: Statistical summary or error message if analysis fails.
    """
    try:
        # Read CSV content into a pandas DataFrame
        df = pd.read_csv(io.StringIO(file_content))
        if df.empty:
            return "Error: The CSV file is empty."
        
        # Basic analysis
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.any():
            return "Error: No numeric columns found for statistical analysis."
        
        stats = df[numeric_cols].describe().to_string()
        return f"CSV Analysis:\n\n{stats}"
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}. Ensure the input is valid CSV format."

@safe_tool
def execute_python_code(code: str) -> str:
    """
    Safely executes a Python code snippet and returns the output.
    
    Args:
        code (str): Python code to execute.
    
    Returns:
        str: Output of the code or error message if execution fails.
    """
    try:
        # Restricted environment for safe execution
        safe_globals = {"__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
        }}
        output = []
        def capture_print(*args, **kwargs):
            output.append(" ".join(map(str, args)))
        
        safe_globals["__builtins__"]["print"] = capture_print
        exec(code, safe_globals, {})
        return "\n".join(output) if output else "Code executed successfully (no output)."
    except Exception as e:
        return f"Error executing Python code: {str(e)}. Ensure the code is valid and uses supported functions."

@safe_tool
def generate_regex_match(pattern: str, text: str) -> str:
    """
    Applies a regex pattern to text and returns matching results.
    
    Args:
        pattern (str): Regular expression pattern.
        text (str): Text to search for matches.
    
    Returns:
        str: Matching results or error message if pattern is invalid.
    """
    try:
        regex = re.compile(pattern)
        matches = regex.findall(text)
        if not matches:
            return f"No matches found for pattern '{pattern}' in the provided text."
        return f"Matches for pattern '{pattern}':\n{', '.join(str(m) for m in matches)}"
    except re.error as e:
        return f"Error in regex pattern: {str(e)}. Please provide a valid regex pattern."

def get_agent_tools() -> List[Tool]:
    """
    Provides a comprehensive set of tools with enhanced error handling and detailed descriptions.
    
    Returns:
        List[Tool]: List of initialized LangChain Tool objects.
    """
    tools = []
    
    # Web Search Tool
    try:
        search_tool = Tool(
            name="web_search",
            func=DuckDuckGoSearchRun().run,
            description=(
                "Performs a web search for real-time information, current events, or unknown topics. "
                "Input: A search query (e.g., 'latest space discoveries'). "
                "Output: A summary of relevant web results."
            )
        )
        tools.append(search_tool)
        logger.info("Added web search tool")
    except Exception as e:
        logger.error(f"Failed to initialize web search tool: {e}")
    
    # Wikipedia Tool
    try:
        wikipedia_tool = Tool(
            name="wikipedia",
            func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2)).run,
            description=(
                "Queries Wikipedia for factual information about people, places, events, or concepts. "
                "Input: A search query (e.g., 'Apollo 11 mission'). "
                "Output: A concise summary from Wikipedia."
            )
        )
        tools.append(wikipedia_tool)
        logger.info("Added Wikipedia tool")
    except Exception as e:
        logger.error(f"Failed to initialize Wikipedia tool: {e}")
    
    # Current Time Tool
    time_tool = Tool(
        name="current_time",
        func=get_current_time,
        description=(
            "Retrieves the current date and time in a specified timezone. "
            "Input: A timezone name (e.g., 'UTC', 'America/New_York'). "
            "Output: Formatted date and time string."
        )
    )
    tools.append(time_tool)
    logger.info("Added current time tool")
    
    # Calculator Tool
    calc_tool = Tool(
        name="calculator",
        func=calculate,
        description=(
            "Evaluates mathematical expressions for arithmetic calculations. "
            "Input: A mathematical expression (e.g., '2 + 2', '(5 * 3) / 2'). "
            "Output: The calculated result."
        )
    )
    tools.append(calc_tool)
    logger.info("Added calculator tool")
    
    # Weather Tool
    weather_tool = Tool(
        name="weather",
        func=get_weather,
        description=(
            "Fetches current weather information for a specified city using OpenWeatherMap API. "
            "Input: A city name (e.g., 'London', 'Tokyo'). "
            "Output: Weather conditions, temperature, humidity, and wind speed."
        )
    )
    tools.append(weather_tool)
    logger.info("Added weather tool")
    
    # Currency Converter Tool
    currency_tool = Tool(
        name="currency_converter",
        func=lambda x: convert_currency(*x.split(",")) if "," in x else "Error: Input must be 'amount,from_currency,to_currency'",
        description=(
            "Converts an amount from one currency to another using real-time exchange rates. "
            "Input: Comma-separated amount, source currency, and target currency (e.g., '100,USD,EUR'). "
            "Output: Converted amount."
        )
    )
    tools.append(currency_tool)
    logger.info("Added currency converter tool")
    
    # CSV Analysis Tool
    csv_tool = Tool(
        name="csv_analyzer",
        func=analyze_csv,
        description=(
            "Analyzes CSV content and provides statistical insights for numeric columns. "
            "Input: Raw CSV content as a string. "
            "Output: Statistical summary (count, mean, std, min, max, etc.)."
        )
    )
    tools.append(csv_tool)
    logger.info("Added CSV analyzer tool")
    
    # Python Code Execution Tool
    python_tool = Tool(
        name="python_executor",
        func=execute_python_code,
        description=(
            "Safely executes Python code snippets in a restricted environment. "
            "Input: Python code as a string (supports basic operations, print, range, etc.). "
            "Output: Code output or error message."
        )
    )
    tools.append(python_tool)
    logger.info("Added Python code executor tool")
    
    # Regex Matcher Tool
    regex_tool = Tool(
        name="regex_matcher",
        func=lambda x: generate_regex_match(*x.split("|", 1)) if "|" in x else "Error: Input must be 'pattern|text'",
        description=(
            "Applies a regex pattern to text and returns matches. "
            "Input: Pattern and text separated by '|' (e.g., '\d+|Sample text 123'). "
            "Output: List of matches or error message."
        )
    )
    tools.append(regex_tool)
    logger.info("Added regex matcher tool")
    
    return tools
