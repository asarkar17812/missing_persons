"""
Per-100,000-inhabitants yearly missing-persons rate for Mexico.

Originally written by @lapanquecita and bundled with the figshare release of
the RNPDNO dataset; only minor formatting and inline documentation have been
added here. The script consumes the cleaned INEGI cases file and the Mexican
state-population panel, computes the rate of missing persons per 100,000
inhabitants for the last 20 years, and renders the result as a Plotly bar
chart with each bar annotated by rate and absolute count.

Data sources (verbatim from the original):
    Missing-persons data:
        https://consultapublicarnpdno.segob.gob.mx/consulta
    Population data:
        https://datos.gob.mx/busca/dataset/proyecciones-de-la-poblacion-
        de-mexico-y-de-las-entidades-federativas-2020-2070

Both datasets are pre-processed (cleaned, translated, anonymized) before
being consumed here. The bundled population file has had several columns
removed and column headers translated to English.

Font: Montserrat (https://fonts.google.com/specimen/Montserrat).

Run as:
    python scripts/mexico/scaling_per_100k.py
"""

import pandas as pd
import plotly.graph_objects as go


# State-ID lookup. 0 is reserved for "national totals" so the same `main`
# function can render either a single-state plot or the country-wide one.
STATES = {
    0: "Mexico",
    1: "Aguascalientes",
    2: "Baja California",
    3: "Baja California Sur",
    4: "Campeche",
    5: "Coahuila",
    6: "Colima",
    7: "Chiapas",
    8: "Chihuahua",
    9: "Ciudad de México",
    10: "Durango",
    11: "Guanajuato",
    12: "Guerrero",
    13: "Hidalgo",
    14: "Jalisco",
    15: "Estado de México",
    16: "Michoacán",
    17: "Morelos",
    18: "Nayarit",
    19: "Nuevo León",
    20: "Oaxaca",
    21: "Puebla",
    22: "Querétaro",
    23: "Quintana Roo",
    24: "San Luis Potosí",
    25: "Sinaloa",
    26: "Sonora",
    27: "Tabasco",
    28: "Tamaulipas",
    29: "Tlaxcala",
    30: "Veracruz",
    31: "Yucatán",
    32: "Zacatecas",
}


def main(state_id):
    """Build the per-100k rate plot for a single state (or the country).

    Parameters
    ----------
    state_id : int
        Which state to plot. 0 means the national total (no state filter).
        Any value in 1..32 selects a single state (see STATES above).

    Pipeline:
        1. Filter the population file to the requested state and aggregate
           to a yearly total.
        2. Deduplicate the cases file to one row per victim (the source
           data allows multiple report rows per victim).
        3. Filter cases to the requested state if state_id != 0.
        4. Parse DATE_OF_INCIDENCE and DATE_OF_REPORT, then use
           DATE_OF_INCIDENCE preferentially and fall back to DATE_OF_REPORT
           when incidence is missing. This recovers most records that
           would otherwise be dropped due to the 43%-missing
           DATE_OF_INCIDENCE column.
        5. Build a yearly count of cases, attach the matching population,
           compute the per-100k rate, and trim to the most recent 20 years.
        6. Render as a Plotly bar chart with a per-bar rate label and a
           legend summarizing the cumulative total.
    """

    # 1. Population: filter to state, then aggregate to year totals.
    pop = pd.read_csv(r"F:\dsl_CLIMA\projects\submittable\missing persons\source\mexico_missing_persons\population.csv")
    pop = pop[pop["STATE_ID"] == state_id]
    pop = pop.groupby("YEAR").sum(numeric_only=True)

    # 2. Cases: deduplicate to one row per victim.
    df = pd.read_csv(r"F:\dsl_CLIMA\projects\submittable\missing persons\source\mexico_missing_persons\data.csv")
    df = df.groupby("VICTIM_ID").last()

    # 3. Filter to the requested state (state_id == 0 means no filter).
    if state_id != 0:
        df = df[df["STATE_ID"] == state_id]

    # 4. Parse the two date fields and use incidence preferentially.
    df["DATE_OF_INCIDENCE"] = pd.to_datetime(df["DATE_OF_INCIDENCE"], errors="coerce")
    df["DATE_OF_REPORT"] = pd.to_datetime(df["DATE_OF_REPORT"], errors="coerce")
    df["DATE_OF_INCIDENCE"] = df["DATE_OF_INCIDENCE"].fillna(df["DATE_OF_REPORT"])

    # 5. Yearly counts -> rate per 100,000 inhabitants -> last 20 years.
    df = df["DATE_OF_INCIDENCE"].value_counts().resample("YS").sum().to_frame("total")
    df.index = df.index.year
    df["pop"] = pop["POPULATION"]
    df["rate"] = df["total"] / df["pop"] * 100000
    df = df.tail(20)

    # Per-bar text: rate (large, bold) above the absolute count.
    df["text"] = df.apply(
        lambda x: f"<b>{x['rate']:,.2f}</b><br>({x['total']:,.0f})", axis=1
    )

    # 6. Render the bar chart. Color scale runs from yellow (0) to red (max).
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["rate"],
            text=df["text"],
            name=f"Cummulative total: <b>{df['total'].sum():,.0f}</b> actively missing.<br>Doesn't include confidential records.",
            textposition="outside",
            marker_color=df["rate"],
            marker_colorscale="portland",
            marker_cmid=0,
            marker_line_width=0,
            textfont_size=30,
        )
    )

    fig.update_xaxes(
        ticks="outside",
        ticklen=10,
        zeroline=False,
        tickcolor="#FFFFFF",
        linewidth=2,
        showline=True,
        showgrid=True,
        gridwidth=0.35,
        mirror=True,
        nticks=len(df) + 1,
    )

    # Dynamically size the y-axis so the tallest bar's outside-label fits.
    fig.update_yaxes(
        title="Rate per 100,000 inhabitants",
        range=[0, df["rate"].max() * 1.1],
        ticks="outside",
        separatethousands=True,
        tickfont_size=14,
        ticklen=10,
        title_standoff=15,
        tickcolor="#FFFFFF",
        linewidth=2,
        gridwidth=0.35,
        showline=True,
        nticks=20,
        zeroline=False,
        mirror=True,
    )

    fig.update_layout(
        showlegend=True,
        legend_borderwidth=1,
        legend_bordercolor="#FFFFFF",
        legend_x=0.01,
        legend_y=0.98,
        legend_xanchor="left",
        legend_yanchor="top",
        width=1920,
        height=1080,
        font_family="Montserrat",
        font_color="#FFFFFF",
        font_size=24,
        title_text=f"Evolution of the rate of missing and unaccounted-for people in <b>{STATES[state_id]}</b> ({df.index.min()}-{df.index.max()})",
        title_x=0.5,
        title_y=0.965,
        margin_t=80,
        margin_r=40,
        margin_b=120,
        margin_l=130,
        title_font_size=34,
        paper_bgcolor="#2B2B2B",
        plot_bgcolor="#171010",
        annotations=[
            dict(
                x=0.01,
                y=-0.11,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                text="Source: RNPDNO (July 2025)",
            ),
            dict(
                x=0.5,
                y=-0.11,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="top",
                text="Year of incidence",
            ),
            dict(
                x=1.01,
                y=-0.11,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                text="Source: @lapanquecita",
            ),
        ],
    )

    # Filename is the state_id so the 33 possible outputs don't collide.
    fig.write_image(fr"F:\dsl_CLIMA\projects\submittable\missing persons\plots\mexico\{state_id}.png")


if __name__ == "__main__":
    main(0)
