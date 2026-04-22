# 交易系统的日期工具函数
from datetime import datetime, timedelta, date
import pandas as pd
from typing import Optional, Tuple, List
import calendar


class DateUtils:
    """交易中日期操作的实用类。"""

    @staticmethod
    def get_trading_days(
        start_date: str, end_date: str, include_weekends: bool = False
    ) -> List[date]:
        """获取两个日期之间的交易日列表。

        Args:
            start_date: 开始日期，格式为 'YYYY-MM-DD'。
            end_date: 结束日期，格式为 'YYYY-MM-DD'。
            include_weekends: 是否包含周末。

        Returns:
            交易日列表。
        """
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()

        date_range = pd.date_range(start=start, end=end, freq="B" if not include_weekends else "D")
        return [d.date() for d in date_range]

    @staticmethod
    def is_trading_day(check_date: str) -> bool:
        """检查日期是否为交易日（工作日）。

        Args:
            check_date: 要检查的日期，格式为 'YYYY-MM-DD'。

        Returns:
            如果是交易日返回 True，否则返回 False。
        """
        check_date = pd.to_datetime(check_date)
        return check_date.weekday() < 5  # Monday=0, Sunday=6

    @staticmethod
    def get_next_trading_day(from_date: str, n: int = 1) -> date:
        """获取下一个交易日。

        Args:
            from_date: 起始日期，格式为 'YYYY-MM-DD'。
            n: 向前的交易日数量。

        Returns:
            下一个交易日。
        """
        from_date = pd.to_datetime(from_date)
        trading_days = pd.date_range(start=from_date, periods=n + 5, freq="B")
        return trading_days[n - 1].date()

    @staticmethod
    def get_previous_trading_day(from_date: str, n: int = 1) -> date:
        """获取前一个交易日。

        Args:
            from_date: 起始日期，格式为 'YYYY-MM-DD'。
            n: 向后的交易日数量。

        Returns:
            前一个交易日。
        """
        from_date = pd.to_datetime(from_date)
        trading_days = pd.date_range(end=from_date, periods=n + 5, freq="B")
        return trading_days[-n].date()

    @staticmethod
    def get_month_end(date_str: str) -> date:
        """获取给定日期所在月份的最后一天。

        Args:
            date_str: 日期，格式为 'YYYY-MM-DD'。

        Returns:
            月份的最后一天。
        """
        dt = pd.to_datetime(date_str)
        year, month = dt.year, dt.month
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, last_day)

    @staticmethod
    def get_quarter_end(date_str: str) -> date:
        """获取给定日期所在季度的最后一天。

        Args:
            date_str: 日期，格式为 'YYYY-MM-DD'。

        Returns:
            季度的最后一天。
        """
        dt = pd.to_datetime(date_str)
        quarter = (dt.month - 1) // 3 + 1
        month = quarter * 3
        year = dt.year
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, last_day)

    @staticmethod
    def get_year_end(date_str: str) -> date:
        """获取给定日期所在年份的最后一天。

        Args:
            date_str: 日期，格式为 'YYYY-MM-DD'。

        Returns:
            年份的最后一天。
        """
        dt = pd.to_datetime(date_str)
        return date(dt.year, 12, 31)

    @staticmethod
    def split_by_year(
        start_date: str, end_date: str
    ) -> List[Tuple[str, str]]:
        """按年份分割日期范围。

        Args:
            start_date: 开始日期，格式为 'YYYY-MM-DD'。
            end_date: 结束日期，格式为 'YYYY-MM-DD'。

        Returns:
            (年份开始, 年份结束) 元组列表。
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
        """按月份分割日期范围。

        Args:
            start_date: 开始日期，格式为 'YYYY-MM-DD'。
            end_date: 结束日期，格式为 'YYYY-MM-DD'。

        Returns:
            (月份开始, 月份结束) 元组列表。
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
        """将日期范围分割为指定大小的块。

        Args:
            start_date: 开始日期，格式为 'YYYY-MM-DD'。
            end_date: 结束日期，格式为 'YYYY-MM-DD'。
            chunk_size: 每个块的天数。

        Returns:
            (块开始, 块结束) 元组列表。
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
        """获取给定年份的美国市场假日。

        Args:
            year: 要获取假日的年份。

        Returns:
            假日日期列表。
        """
        # 这是美国市场假日的简化列表
        # 在生产环境中，请使用合适的假日日历
        holidays = []

        # 元旦
        try:
            holidays.append(date(year, 1, 1))
        except ValueError:
            pass  # 处理1月1日落在周末的情况

        # 马丁·路德·金纪念日（一月的第三个星期一）
        mlk_day = DateUtils._get_nth_weekday(year, 1, 0, 3)  # 0 = Monday
        holidays.append(mlk_day)

        # 总统日（二月的第三个星期一）
        presidents_day = DateUtils._get_nth_weekday(year, 2, 0, 3)
        holidays.append(presidents_day)

        # 耶稣受难日（近似值 - 复活节前的星期五）
        good_friday = DateUtils._calculate_good_friday(year)
        holidays.append(good_friday)

        # 阵亡将士纪念日（五月的最后一个星期一）
        memorial_day = DateUtils._get_last_weekday(year, 5, 0)
        holidays.append(memorial_day)

        # 六月节（6月19日）
        holidays.append(date(year, 6, 19))

        # 独立日（7月4日）
        holidays.append(date(year, 7, 4))

        # 劳动节（九月的第一个星期一）
        labor_day = DateUtils._get_nth_weekday(year, 9, 0, 1)
        holidays.append(labor_day)

        # 感恩节（十一月的第四个星期四）
        thanksgiving = DateUtils._get_nth_weekday(year, 11, 3, 4)  # 3 = Thursday
        holidays.append(thanksgiving)

        # 圣诞节（12月25日）
        holidays.append(date(year, 12, 25))

        return holidays

    @staticmethod
    def _get_nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """获取给定月份的第n个工作日。

        Args:
            year: 年份。
            month: 月份 (1-12)。
            weekday: 工作日 (0=星期一, 6=星期日)。
            n: 第n次出现 (1=第一次, 4=第四次, 等等)。

        Returns:
            第n个工作日的日期。
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
        """获取给定月份的最后工作日。

        Args:
            year: 年份。
            month: 月份 (1-12)。
            weekday: 工作日 (0=星期一, 6=星期日)。

        Returns:
            最后工作日的日期。
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
        """计算给定年份的耶稣受难日。

        Args:
            year: 年份。

        Returns:
            耶稣受难日日期。
        """
        # 简化计算（Meeus/Jones/Butcher算法）
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