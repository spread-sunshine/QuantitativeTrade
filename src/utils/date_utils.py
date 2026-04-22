# Date utility functions for trading system
from datetime import datetime, timedelta, date
import pandas as pd
from typing import Optional, Tuple, List
import calendar


class DateUtils:
    """Utility class for date operations in trading."""

    @staticmethod
    def get_trading_days(
        start_date: str, end_date: str, include_weekends: bool = False
    ) -> List[date]:
        """Get list of trading days between two dates.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            include_weekends: Whether to include weekends.

        Returns:
            List of trading days.
        """
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()

        date_range = pd.date_range(start=start, end=end, freq="B" if not include_weekends else "D")
        return [d.date() for d in date_range]

    @staticmethod
    def is_trading_day(check_date: str) -> bool:
        """Check if a date is a trading day (weekday).

        Args:
            check_date: Date to check in 'YYYY-MM-DD' format.

        Returns:
            True if it's a trading day, False otherwise.
        """
        check_date = pd.to_datetime(check_date)
        return check_date.weekday() < 5  # Monday=0, Sunday=6

    @staticmethod
    def get_next_trading_day(from_date: str, n: int = 1) -> date:
        """Get the next trading day.

        Args:
            from_date: Starting date in 'YYYY-MM-DD' format.
            n: Number of trading days ahead.

        Returns:
            Next trading day.
        """
        from_date = pd.to_datetime(from_date)
        trading_days = pd.date_range(start=from_date, periods=n + 5, freq="B")
        return trading_days[n - 1].date()

    @staticmethod
    def get_previous_trading_day(from_date: str, n: int = 1) -> date:
        """Get the previous trading day.

        Args:
            from_date: Starting date in 'YYYY-MM-DD' format.
            n: Number of trading days back.

        Returns:
            Previous trading day.
        """
        from_date = pd.to_datetime(from_date)
        trading_days = pd.date_range(end=from_date, periods=n + 5, freq="B")
        return trading_days[-n].date()

    @staticmethod
    def get_month_end(date_str: str) -> date:
        """Get the last day of the month for a given date.

        Args:
            date_str: Date in 'YYYY-MM-DD' format.

        Returns:
            Last day of the month.
        """
        dt = pd.to_datetime(date_str)
        year, month = dt.year, dt.month
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, last_day)

    @staticmethod
    def get_quarter_end(date_str: str) -> date:
        """Get the last day of the quarter for a given date.

        Args:
            date_str: Date in 'YYYY-MM-DD' format.

        Returns:
            Last day of the quarter.
        """
        dt = pd.to_datetime(date_str)
        quarter = (dt.month - 1) // 3 + 1
        month = quarter * 3
        year = dt.year
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, last_day)

    @staticmethod
    def get_year_end(date_str: str) -> date:
        """Get the last day of the year for a given date.

        Args:
            date_str: Date in 'YYYY-MM-DD' format.

        Returns:
            Last day of the year.
        """
        dt = pd.to_datetime(date_str)
        return date(dt.year, 12, 31)

    @staticmethod
    def split_by_year(
        start_date: str, end_date: str
    ) -> List[Tuple[str, str]]:
        """Split date range by year.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            List of (year_start, year_end) tuples.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        periods = []
        current_year = start.year

        while current_year <= end.year:
            year_start = max(start, pd.Timestamp(f"{current_year}-01-01"))
            year_end = min(end, pd.Timestamp(f"{current_year}-12-31"))

            if year_start <= year_end:
                periods.append(
                    (year_start.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d"))
                )

            current_year += 1

        return periods

    @staticmethod
    def split_by_month(
        start_date: str, end_date: str
    ) -> List[Tuple[str, str]]:
        """Split date range by month.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            List of (month_start, month_end) tuples.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        periods = []
        current = start

        while current <= end:
            month_start = current.replace(day=1)
            month_end = (current + pd.offsets.MonthEnd(0)).replace(
                hour=23, minute=59, second=59
            )

            if month_start < start:
                month_start = start

            if month_end > end:
                month_end = end

            if month_start <= month_end:
                periods.append(
                    (month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))
                )

            current = month_end + pd.Timedelta(days=1)

        return periods

    @staticmethod
    def get_date_ranges(
        start_date: str, end_date: str, chunk_size: int = 30
    ) -> List[Tuple[str, str]]:
        """Split date range into chunks of specified size.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            chunk_size: Number of days per chunk.

        Returns:
            List of (chunk_start, chunk_end) tuples.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        periods = []
        current = start

        while current <= end:
            chunk_end = min(current + pd.Timedelta(days=chunk_size - 1), end)
            periods.append(
                (current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            )
            current = chunk_end + pd.Timedelta(days=1)

        return periods

    @staticmethod
    def get_market_holidays(year: int) -> List[date]:
        """Get US market holidays for a given year.

        Args:
            year: Year to get holidays for.

        Returns:
            List of holiday dates.
        """
        # This is a simplified list of US market holidays
        # In production, use a proper holiday calendar
        holidays = []

        # New Year's Day
        try:
            holidays.append(date(year, 1, 1))
        except ValueError:
            pass  # Handle cases where Jan 1 falls on weekend

        # Martin Luther King Jr. Day (third Monday in January)
        mlk_day = DateUtils._get_nth_weekday(year, 1, 0, 3)  # 0 = Monday
        holidays.append(mlk_day)

        # Presidents Day (third Monday in February)
        presidents_day = DateUtils._get_nth_weekday(year, 2, 0, 3)
        holidays.append(presidents_day)

        # Good Friday (approximation - Friday before Easter)
        good_friday = DateUtils._calculate_good_friday(year)
        holidays.append(good_friday)

        # Memorial Day (last Monday in May)
        memorial_day = DateUtils._get_last_weekday(year, 5, 0)
        holidays.append(memorial_day)

        # Juneteenth (June 19)
        holidays.append(date(year, 6, 19))

        # Independence Day (July 4)
        holidays.append(date(year, 7, 4))

        # Labor Day (first Monday in September)
        labor_day = DateUtils._get_nth_weekday(year, 9, 0, 1)
        holidays.append(labor_day)

        # Thanksgiving Day (fourth Thursday in November)
        thanksgiving = DateUtils._get_nth_weekday(year, 11, 3, 4)  # 3 = Thursday
        holidays.append(thanksgiving)

        # Christmas Day (December 25)
        holidays.append(date(year, 12, 25))

        return holidays

    @staticmethod
    def _get_nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """Get the nth weekday of a given month.

        Args:
            year: Year.
            month: Month (1-12).
            weekday: Weekday (0=Monday, 6=Sunday).
            n: Nth occurrence (1=first, 4=fourth, etc.)

        Returns:
            Date of the nth weekday.
        """
        if n < 1 or n > 5:
            raise ValueError("n must be between 1 and 5")

        # Get the first day of the month
        first_day = date(year, month, 1)
        first_weekday = first_day.weekday()

        # Calculate the first occurrence of the target weekday
        days_to_add = (weekday - first_weekday) % 7
        first_occurrence = first_day + timedelta(days=days_to_add)

        # Add (n-1) weeks
        target_date = first_occurrence + timedelta(weeks=n - 1)

        # Check if we're still in the same month
        if target_date.month != month:
            raise ValueError(f"Month {month} doesn't have {n} occurrences of weekday {weekday}")

        return target_date

    @staticmethod
    def _get_last_weekday(year: int, month: int, weekday: int) -> date:
        """Get the last weekday of a given month.

        Args:
            year: Year.
            month: Month (1-12).
            weekday: Weekday (0=Monday, 6=Sunday).

        Returns:
            Date of the last weekday.
        """
        # Get the last day of the month
        last_day = DateUtils.get_month_end(f"{year}-{month:02d}-01")
        last_weekday = last_day.weekday()

        # Calculate days to subtract
        days_to_subtract = (last_weekday - weekday) % 7
        if days_to_subtract == 0:
            days_to_subtract = 7

        return last_day - timedelta(days=days_to_subtract - 1)

    @staticmethod
    def _calculate_good_friday(year: int) -> date:
        """Calculate Good Friday for a given year.

        Args:
            year: Year.

        Returns:
            Good Friday date.
        """
        # Simplified calculation (Meeus/Jones/Butcher algorithm)
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1

        easter = date(year, month, day)
        good_friday = easter - timedelta(days=2)

        return good_friday