import subprocess
import time


def get_upload_speed(msg=None):
    result = subprocess.run(
        ['speedtest-cli', '--no-download', '--bytes'],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.split('\n'):
        if "Upload" in line:
            tokens = line.split()
            upload_speed = float(tokens[1])
            unit = tokens[2]
            upload_speed = float(line.split()[1])
            if msg is not None:
                print(f'{msg}: {upload_speed:.2f} {unit}')
            if 'Mbyte' in unit:
                return upload_speed * 1e6
            elif 'Kbyte' in unit:
                return upload_speed * 1e3
            elif 'Gbyte' in unit:
                return upload_speed * 1e9
            else:
                return upload_speed
    return None


def compute_speed_statistics(iterations=10):
    speeds = []
    for i in range(iterations):
        speed = get_upload_speed(f'Iteration {i+1}')
        if speed:
            speeds.append(speed)

        # small sleep to avoid server overwhelming
        time.sleep(1)

    min_speed = min(speeds)
    max_speed = max(speeds)
    avg_speed = sum(speeds) / len(speeds)

    return min_speed, max_speed, avg_speed


def int_to_human(speed):
    def between(x, low, high):
        return low <= x < high

    if between(speed, 1e9, 1e12):
        return f'{speed/1e9:.2f} Gbyte/sec'
    elif between(speed, 1e6, 1e9):
        return f'{speed/1e6:.2f} Mbyte/sec'
    elif between(speed, 1e3, 1e6):
        return f'{speed/1e3:.2f} Kbyte/sec'
    else:
        return f'{speed:.2f} byte/sec'


def write_statistics_to_file(filename, min_speed, max_speed, avg_speed):
    ith = int_to_human
    with open(filename, 'w') as file:
        file.write(f"Minimum Upload Speed: {ith(min_speed)}\n")
        file.write(f"Maximum Upload Speed: {ith(max_speed)}\n")
        file.write(f"Average Upload Speed: {ith(avg_speed)}\n")


def main():
    min_speed, max_speed, avg_speed = compute_speed_statistics()
    write_statistics_to_file(
        'upload_speed_statistics.txt', min_speed, max_speed, avg_speed
    )


if __name__ == "__main__":
    main()
