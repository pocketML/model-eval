from time import time, sleep
import curses

# Utility methods for formatting stuff.
def zero_pad(num):
    if num < 10:
        return "0" + str(num)
    return str(num)

def format_time(seconds):
    hours = 0
    minutes = 0
    if seconds > 60:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
    if minutes > 60:
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
        prog_str = "#" * curr_ticks
        remain_str = " " * (self.total_ticks - curr_ticks)
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

        loadbar_width = ((self.total_ticks + len(str(self.total_steps)) + 14))
        padding = 6

        screen = self.screen.subwin(5, loadbar_width + (padding * 2), 0, 0)
        screen.clear()
        screen.box()

        start_x = padding
        header_x = max(int((loadbar_width / 2) - (len(header) / 2)), 0)

        screen.addstr(1, start_x + header_x, header)
        str_2 = f"[{prog_str}{remain_str}] ({pct}% | {self.curr_step}/{self.total_steps})"
        str_3 = f"[{spent_str} < {left_str} | {items_avg:.2f} it/s]"
        screen.addstr(2, start_x, str_2)
        prog_x = max(int((loadbar_width / 2) - (len(str_3) / 2)), 0)
        screen.addstr(3, start_x + prog_x, str_3)
        screen.refresh()

    def reset_console(self):
        self.screen.clear()
        self.screen.keypad(False)
        curses.echo()
        curses.nocbreak()
        curses.endwin()

    def __iter__(self):
        return LoadbarIterator(self)

if __name__ == "__main__":
    load = Loadbar(30, 5, "Training... ")
    for i in range(5):
        accuracy = i * 10
        load.step(text=f"Test Acc: {accuracy}%")
        sleep(0.5)
