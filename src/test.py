import curses

print("TEST123")
print("TEST123")
print("32131")
print("sda")
print("TEST123")
print("3213124a")

curses.initscr()
curses.noecho()
curses.cbreak()

def do_stuff(scr):
    scr.addstr(0, 5, "Test")
    scr.addstr(0, 5, "Test")
    scr.refresh()
    scr.getkey()

curses.wrapper(do_stuff)
