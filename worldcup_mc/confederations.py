"""
Team -> FIFA confederation lookup.

Six confederations:
  UEFA      Europe
  CONMEBOL  South America
  CONCACAF  North/Central America + Caribbean
  CAF       Africa
  AFC       Asia + Australia
  OFC       Oceania

Used by the rating fit to partial-pool each team toward its CONFEDERATION's
mean rather than the global mean, so "best team in a weak region" is not
mistaken for "strong in absolute terms". The cross-confederation games that
exist (World Cups, intercontinental playoffs, friendlies) pin the
confederations to a common scale; the pooling spreads that signal to teams
that rarely leave their region.

Names follow the martj42 dataset spelling. Teams not found here fall back to
confederation "OTHER" and are pooled toward the global mean (old behaviour),
so a missing minnow never breaks the fit -- but add any you care about.
"""
from __future__ import annotations

CONFEDERATION: dict[str, str] = {}


def _add(conf: str, teams: list[str]) -> None:
    for t in teams:
        CONFEDERATION[t] = conf


_add("UEFA", [
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
    "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Czechia", "Denmark", "England", "Estonia",
    "Faroe Islands", "Finland", "France", "Georgia", "Germany", "Gibraltar",
    "Greece", "Hungary", "Iceland", "Republic of Ireland", "Ireland",
    "Israel", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Malta", "Moldova", "Montenegro",
    "Netherlands", "North Macedonia", "Northern Ireland", "Norway", "Poland",
    "Portugal", "Romania", "Russia", "San Marino", "Scotland", "Serbia",
    "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey",
    "Ukraine", "Wales",
])

_add("CONMEBOL", [
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Paraguay", "Peru", "Uruguay", "Venezuela",
])

_add("CONCACAF", [
    "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
    "Belize", "Bermuda", "British Virgin Islands", "Canada", "Cayman Islands",
    "Costa Rica", "Cuba", "Curaçao", "Dominica", "Dominican Republic",
    "El Salvador", "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras",
    "Jamaica", "Mexico", "Montserrat", "Nicaragua", "Panama", "Puerto Rico",
    "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago",
    "Turks and Caicos Islands", "United States", "US Virgin Islands",
])

_add("CAF", [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Congo", "DR Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea",
    "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
    "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco",
    "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
    "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone",
    "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo",
    "Tunisia", "Uganda", "Zambia", "Zimbabwe",
])

_add("AFC", [
    "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei",
    "Cambodia", "China PR", "China", "Chinese Taipei", "Guam", "Hong Kong",
    "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan", "Kuwait",
    "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia", "Maldives",
    "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan",
    "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore",
    "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Thailand",
    "Timor-Leste", "Turkmenistan", "United Arab Emirates", "Uzbekistan",
    "Vietnam", "Yemen",
])

_add("OFC", [
    "American Samoa", "Cook Islands", "Fiji", "New Caledonia", "New Zealand",
    "Papua New Guinea", "Samoa", "Solomon Islands", "Tahiti", "Tonga",
    "Vanuatu",
])


def confederation_of(team: str) -> str:
    """Confederation for a team, or 'OTHER' if unknown (pooled to global mean)."""
    return CONFEDERATION.get(team, "OTHER")


def all_confederations() -> list[str]:
    return ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC", "OTHER"]
