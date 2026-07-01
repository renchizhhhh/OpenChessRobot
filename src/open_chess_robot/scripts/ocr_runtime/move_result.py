"""Pure helpers for the /chess_move_res acknowledgement protocol.

The robot node publishes a result string for every move it executes; the
commander matches that string against the move it is currently waiting on. This
module holds the format and matching rules with no ROS dependency so they can be
unit tested and kept consistent between publisher and subscriber.

Result format: ``"<encoded_move> <status>"`` where ``<encoded_move>`` is the UCI
move (4 chars, or 5 for a promotion) followed by five binary flag digits, and
``<status>`` is ``FINISHED_SUFFIX`` on success or ``FAILED_SUFFIX`` on failure.
"""

FINISHED_SUFFIX = "is finished"
FAILED_SUFFIX = "failed"


def encode_result(encoded_move, success=True):
    """Build the /chess_move_res string for a completed (or failed) move."""
    suffix = FINISHED_SUFFIX if success else FAILED_SUFFIX
    return f"{encoded_move} {suffix}"


def result_matches(last_move_send, result):
    """True if ``result`` acknowledges the command ``last_move_send``.

    ``last_move_send`` is the raw UCI move; the result begins with that UCI
    (then flag digits and a status), so a prefix match works for both 4-char and
    5-char (promotion) moves and rejects results for any other move. An empty
    pending move never matches.
    """
    if not last_move_send:
        return False
    return result.startswith(last_move_send)


def result_is_failure(result):
    """True if ``result`` reports a failed execution rather than success."""
    return result.rstrip().endswith(FAILED_SUFFIX)
