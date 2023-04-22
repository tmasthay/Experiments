# Prisoners with Red and Blue Hats Simulator

This is a simulator for the classic "Prisoners with red and blue hats" logic puzzle. It helps you visualize and understand the solution to the problem by simulating different scenarios with varying number of people in line.

## Table of Contents
- [Usage](#usage)
- [Options](#options)
- [Example](#example)

## Usage
To run the simulator, use the following command:

```
python main.py [options]
```

## Options
The following command-line options are available:

| Option   | Description                       |
|----------|-----------------------------------|
| `-h`, `--help`   | Show the help message and exit. |
| `--N N`  | Set the number of people in line. Defaults to 5. |
| `--verbose` | Show verbose output.          |
| `--slow` | Show verbose output stepwise. Press enter to continue output. If `verbose` is set to `False` and `slow` to `True`, the output will still show.             |

## Example
To run the simulator with 10 people in line and verbose output, use the following command:

```
python main.py --N 10 --verbose
```

Enjoy exploring the "Prisoners with red and blue hats" logic puzzle with this simulator!
