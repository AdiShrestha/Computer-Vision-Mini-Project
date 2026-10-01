"""Actual child-process resource observations; unsupported measurements are null.

RSS is an OS-reported child process high-water mark, not total system, unified
or accelerator memory. wait4 uses per-child accounting rather than the running
maximum across previously waited children.
"""
import os
import platform
import subprocess
import time


def observed_run(argv, *, cwd, env, stdout, stderr, preexec_fn=None):
    start = time.monotonic()
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                               preexec_fn=preexec_fn)
    cpu = peak = None
    method = 'wall_only'
    reason = 'PER_CHILD_RUSAGE_UNAVAILABLE'
    try:
        if hasattr(os, 'wait4'):
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            cpu = float(usage.ru_utime + usage.ru_stime)
            system = platform.system()
            if system in ('Darwin', 'Linux'):
                peak = int(usage.ru_maxrss) * (1 if system == 'Darwin' else 1024)
                reason = None
            else:
                reason = 'UNKNOWN_MAXRSS_UNITS'
            method = 'wait4_child_rusage'
        else:
            process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    return process.returncode, {
        'wall_time_seconds': time.monotonic()-start,
        'cpu_time_seconds': cpu, 'memory_peak_bytes': peak,
        'measurement_method': method, 'unavailable_reason': reason,
        'memory_scope': 'OS child-process RSS high-water mark; excludes total unified/GPU memory',
    }
