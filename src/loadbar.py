from time import time, sleep
import curses

ASCII_LOGO_TRANSPARENT = [
    "                           _|                 _|      _|      _| _|      ",
    "_|_|_|     _|_|     _|_|_| _|  _|    _|_|   _|_|_|_|  _|_|  _|_| _|      ",
    "_|    _| _|    _| _|       _|_|    _|_|_|_|   _|      _|  _|  _| _|      ",
    "_|    _| _|    _| _|       _|  _|  _|         _|      _|      _| _|      ",
    "_|_|_|     _|_|     _|_|_| _|    _|  _|_|_|     _|_|  _|      _| _|_|_|_|",
    "_|                                                                       ",
    "_|                                                                       "
]

ASCII_LOGO_FILLED = [
    "                           █|                 █|      █|      █| █|      ",
    "█████|     ███|     █████| █|  █|    ███|   ███████|  ███|  ███| █|      ",
    "█|    █| █|    █| █|       ███|    ███|███|   █|      █|  █|  █| █|      ",
    "█|    █| █|    █| █|       █|  █|  █|         █|      █|      █| █|      ",
    "█████|     ███|     █████| █|    █|  █████|     ███|  █|      █| ███████|",
    "█|                                                                       ",
    "█|                                                                       "
]

# Utility methods for formatting stuff.
def zero_pad(num):
    if num < 10:
        return "0" + str(num)
    return str(num)

def format_time(seconds):
    hours = 0
    minutes = 0
    if seconds >= 60:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
    if minutes >= 60:
        hours = int(minutes / 60)
        minutes = int(minutes % 60)
    mins_hours_str = f"{zero_pad(minutes)}:{zero_pad(seconds)}"
    if hours > 0:
        return f"{zero_pad(hours)}:{mins_hours_str}"
    return mins_hours_str

class LoadbarIterator:
    def __init__(self, loadbar):
        self.loadbar = loadbar

    def __next__(self):
        num = self.loadbar.curr_step
        self.loadbar.step()
        if self.loadbar.curr_step == self.loadbar.total_steps:
            raise StopIteration
        return num

class Loadbar:
    def __init__(self, total_ticks, total_steps, text="", curr_step=0):
        self.total_ticks = total_ticks
        self.total_steps = total_steps
        self.curr_step = curr_step
        self.desc = text
        self.curr_ratio = curr_step / total_steps
        self.curr_tick = self.total_ticks * self.curr_ratio
        self.time_start = time()
        self.last_time = time()
        self.last_second = self.last_time
        self.items_per_second = [0]
        self.last_ticks = [0]

        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()

    def step(self, amount=1, text=""):
        if len(self.last_ticks) == 1:
            self.print_bar()
        self.curr_step += amount
        prev_pct = int(self.curr_ratio * 100)
        self.curr_ratio = self.curr_step / self.total_steps

        time_now = time()
        self.last_ticks.append(time_now - self.last_time)

        if len(self.last_ticks) > 10:
            self.last_ticks.pop(0)

        self.items_per_second[-1] += 1

        if time_now - self.last_second > 1:
            self.last_second = time_now
            self.items_per_second.append(0)
            if len(self.items_per_second) == 10:
                self.items_per_second.pop(0)

        pct = int(self.curr_ratio * 100)
        if pct > prev_pct:
            tick = int(self.curr_ratio * self.total_ticks)
            self.curr_tick = tick
            if self.curr_ratio > 1:
                self.curr_ratio = 1
                self.curr_tick = self.total_steps

            self.print_bar(text)
            if self.curr_step == self.total_steps:
                self.reset_console()
        self.last_time = time()

    def print_bar(self, flavor_text=""):
        curr_ticks = int(self.curr_ratio * self.total_ticks)
        prog_str = "■" * curr_ticks
        remain_str = "□" * (self.total_ticks - curr_ticks)
        pct = int(self.curr_ratio * 100)
        time_now = time()
        time_spent = int(time_now - self.time_start)
        avg_time = sum(self.last_ticks) / len(self.last_ticks)
        time_left = int(avg_time * (self.total_steps - self.curr_step))
        items_avg = sum(self.items_per_second) / len(self.items_per_second)

        header = self.desc if self.desc != "" else flavor_text
        if self.desc != "" and flavor_text != "":
            header = header + " - " + flavor_text
        spent_str = format_time(time_spent)
        left_str = format_time(time_left)
        str_2 = f"{prog_str}{remain_str}"
        str_3 = f"[{pct}% | {self.curr_step}/{self.total_steps} | {spent_str} < {left_str} | {items_avg:.2f} it/s]"

        loadbar_width = self.total_ticks
        header_x = max(int((loadbar_width / 2) - (len(header) / 2)), 0)

        padding = 6
        start_x = int((len(ASCII_LOGO_TRANSPARENT[0]) / 2) - (loadbar_width / 2)) + (padding // 2)

        max_x = max(len(x) for x in ASCII_LOGO_TRANSPARENT)
        screen_width = max_x + (padding * 2) - 4
        screen = self.screen.subwin(14, screen_width, 0, 1)
        screen.clear()
        screen.box()

        self._draw_ascii_logo(screen)

        start_y = 9
        screen.addstr(start_y, start_x + header_x, header)

        screen.addstr(start_y + 1, start_x, str_2)
        prog_x = max(int((loadbar_width / 2) - (len(str_3) / 2)), -start_x + 1)
        screen.addstr(start_y + 2, start_x + prog_x, str_3)
        screen.refresh()

    def reset_console(self):
        self.screen.clear()
        self.screen.keypad(False)
        curses.echo()
        curses.nocbreak()
        curses.endwin()

    def _draw_ascii_logo(self, screen):
        start_y = 1
        max_x = max(len(x) for x in ASCII_LOGO_TRANSPARENT)
        x = 0
        y = 0
        squares = 0
        down = True

        num_squares = sum(x.count("_") for x in ASCII_LOGO_TRANSPARENT)

        while x < max_x and squares / num_squares < self.curr_ratio:
            if ASCII_LOGO_TRANSPARENT[y][x] == "_":
                squares += 1

            if down:
                y += 1
            else:
                y -= 1

            if y == 0 or y == len(ASCII_LOGO_FILLED) - 1:
                x += 1
                down = not down

            x * len(ASCII_LOGO_FILLED) + y

        start = 0 if down else len(ASCII_LOGO_TRANSPARENT) - 1
        end = len(ASCII_LOGO_TRANSPARENT) if down else -1
        step = 1 if down else -1

        for i in range(start, end, step):
            index_x = x
            if down and y > i:
                index_x += 1
            elif not down and y < i:
                index_x += 1

            str_filled = ASCII_LOGO_FILLED[i][:index_x]
            str_unfilled = ASCII_LOGO_TRANSPARENT[i][index_x:]

            screen.addstr(start_y + i, 4, str_filled + str_unfilled)

    def __iter__(self):
        return LoadbarIterator(self)

if __name__ == "__main__":
    load = Loadbar(50, 200, "Training... ")
    for i in range(200):
        accuracy = int(i * 0.5)
        load.step(text=f"Test Acc: {accuracy}%")
        sleep(0.1)
