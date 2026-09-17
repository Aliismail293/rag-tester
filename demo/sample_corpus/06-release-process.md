# Release Process and Lunar Build Numbers

Thistlewick Robotics ships firmware updates every 6 weeks using a
scheme it calls Lunar Build Numbers. Each release is named after one of
the invented moons of Velindra, a fictional planet used internally as
a running joke: recent releases include Vel-3.Callix and
Vel-3.Tethis. A release only ships if it passes the full hardware
regression suite twice in a row on the reference EL-7 unit. If a
release fails regression testing, the release train slips by exactly
one week rather than shipping partial fixes.
