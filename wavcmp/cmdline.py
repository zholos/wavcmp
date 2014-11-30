from __future__ import print_function

import sys, argparse

from . import SilenceableError, File, Audio

def main():
    parser = argparse.ArgumentParser(
        prog="wavcmp",
        description="Compare two tracks at different offsets. Return true if "
                    "they match.",
        add_help=False)

    parser.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version="%(prog)s 0.4",
                        help=argparse.SUPPRESS)
    parser.add_argument("-o", metavar="offset", type=float,
                        help="maximum offset, default 0.5 seconds")
    parser.add_argument("-t", metavar="threshold", type=lambda x: float(x)/100.,
                        help="match threshold, default 1%%")
    parser.add_argument("-k", metavar="skip", type=int,
                        help="max skipped album tracks, default 0")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", action="store_true",
                       help="verbose match display")
    group.add_argument("-q", action="store_true",
                       help="don't show match statistics")
    group.add_argument("-M", action="store_true",
                        help="machine-readable output")
    parser.add_argument("-s", action="store_true",
                        help="silently fail on invalid files")
    #parser.add_argument("-w", metavar="write",
    #                    help="write difference for visual inspection")
    parser.add_argument("a", metavar="a.flac") # help=argparse.SUPPRESS
    parser.add_argument("b", metavar="b.flac") # help=argparse.SUPPRESS
    args = parser.parse_args()

    try:
        a, b = map(File, (args.a, args.b))
        for i in (a, b):
            if not isinstance(i, Audio):
                raise SilenceableError(
                    "Not a stereo audio file or directory containing them: "
                    "'{}'".format(i.filename))
        if a.rate != b.rate:
            raise SilenceableError(
                "Sample rates different in files: "
                "'{}' and '{}'".format(a.filename, b.filename))
        matches = a.cmp(b, offset=args.o, threshold=args.t, skip=args.k)

        matched = False
        for match in matches:
            matched = True
            if args.q:
                break
            elif args.M:
                match.show_machine_readable()
                break
            else:
                match.show(verbose=args.v)
        parser.exit(0 if matched else 1)

    except (EnvironmentError, RuntimeError) as e:
        if args.s and isinstance(e, SilenceableError):
            message = None
        else:
            message = "{0}: error: {1}".format(parser.prog, e)
        parser.exit(1, message=message)
