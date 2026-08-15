"""Open-Meteo unit declarations — never infer; always assert."""

from __future__ import annotations

from typing import Any, Iterable


def extract_responses(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise TypeError(f"Unexpected Open-Meteo payload type: {type(payload)!r}")


def assert_hourly_units(
    payload: Any,
    *,
    expected: dict[str, str],
    context: str = "",
) -> None:
    """Fail loudly if the API's declared hourly_units disagree with expectation.

    ``expected`` maps hourly variable name → exact unit string (e.g. ``"m/s"``, ``"kn"``, ``"°"``).
    Only keys present in both ``expected`` and the response's ``hourly_units`` are checked;
    missing expected keys that appear in ``hourly`` data still fail.
    """

    where = f" ({context})" if context else ""
    responses = extract_responses(payload)
    if not responses:
        raise ValueError(f"Open-Meteo unit assertion failed{where}: empty payload")

    for idx, response in enumerate(responses):
        units = response.get("hourly_units") or {}
        hourly = response.get("hourly") or {}
        if not isinstance(units, dict):
            raise ValueError(
                f"Open-Meteo unit assertion failed{where}: "
                f"response[{idx}] missing hourly_units dict"
            )
        for var, want in expected.items():
            if var not in hourly and var not in units:
                continue
            got = units.get(var)
            if got is None:
                raise ValueError(
                    f"Open-Meteo unit assertion failed{where}: "
                    f"response[{idx}] variable {var!r} present but hourly_units lacks it"
                )
            if str(got) != str(want):
                raise ValueError(
                    f"Open-Meteo unit assertion failed{where}: "
                    f"response[{idx}] {var} unit is {got!r}, expected {want!r}. "
                    f"Refusing to convert with the wrong scale."
                )


def assert_speed_unit_ms(payload: Any, variables: Iterable[str], *, context: str = "") -> None:
    expected = {name: "m/s" for name in variables}
    assert_hourly_units(payload, expected=expected, context=context)


def assert_direction_unit_deg(payload: Any, variables: Iterable[str], *, context: str = "") -> None:
    expected = {name: "°" for name in variables}
    assert_hourly_units(payload, expected=expected, context=context)
