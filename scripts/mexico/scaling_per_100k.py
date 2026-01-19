"""

This script creates a bar chart showing the trend of the rate of
missing people for the specified state in Mexico.

It is very easy to use, you only need to run the main function with
one parameter: the ID of the state.

The missing people data was sourced from:

https://consultapublicarnpdno.segob.gob.mx/consulta

The original data was cleaned, translated and anonymized.

The population data was sourced from:

https://datos.gob.mx/busca/dataset/proyecciones-de-la-poblacion-de-mexico-y-de-las-entidades-federativas-2020-2070/resource/ae83f2b0-f23e-45e3-91ae-f85594775dff

The population dataset included in this project is meant to only
be used within the project, as it had some adjustments done (removal of
columns and translation).

This project uses the free Montserrat font -> https://fonts.google.com/specimen/Montserrat

"""

import pandas as pd
import plotly.graph_objects as go


# Each state_id has a corresponding name.
# 0 is for national level figures.
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
    """
    Creates a bar chart with the yearly rate of missing people
    for the specified state.

    Parameters
    ----------
    state_id : int
        The ID of the state you want to plot.
        Use 0 for national figures.

    """

    # We load the population dataset.
    pop = pd.read_csv(r"F:\dsl_CLIMA\projects\submittable\missing persons\source\mexico_missing_persons\population.csv")

    # We filter by the specified state_id.
    pop = pop[pop["STATE_ID"] == state_id]

    # We calculate the total population by year.
    pop = pop.groupby("YEAR").sum(numeric_only=True)

    # We load the missing people dataset.
    df = pd.read_csv(r"F:\dsl_CLIMA\projects\submittable\missing persons\source\mexico_missing_persons\data.csv")

    # A victim can have multiple reports, the first thing to do
    # is to only select one record per victim.
    df = df.groupby("VICTIM_ID").last()

    # We filter by state. If it is 0 we skip this step as
    # 0 is for national level figures.
    if state_id != 0:
        df = df[df["STATE_ID"] == state_id]

    # We will convert to datetime the date of incidence and date of report columns.
    df["DATE_OF_INCIDENCE"] = pd.to_datetime(df["DATE_OF_INCIDENCE"], errors="coerce")
    df["DATE_OF_REPORT"] = pd.to_datetime(df["DATE_OF_REPORT"], errors="coerce")

    # To get the counts by year we will prefer the date of incidence
    # but we don't always have it. In that case we fallback to the date of report.
    df["DATE_OF_INCIDENCE"] = df["DATE_OF_INCIDENCE"].fillna(df["DATE_OF_REPORT"])

    # Now we calculate the yearly counts.
    df = df["DATE_OF_INCIDENCE"].value_counts().resample("YS").sum().to_frame("total")

    # We only need the year from the index.
    df.index = df.index.year

    # We add the population to our missing people DataFrame.
    df["pop"] = pop["POPULATION"]

    # We calculate the rate per 100,000 inhabitants.
    df["rate"] = df["total"] / df["pop"] * 100000

    # We only select the latest 20 years.
    df = df.tail(20)

    # We create the text for each bar.
    df["text"] = df.apply(
        lambda x: f"<b>{x['rate']:,.2f}</b><br>({x['total']:,.0f})", axis=1
    )

    # We will create a simple bar chart with all the previous calculations.
    # The bar chart will have a color scale from yellow (0) to red (max).
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

    # The y-axis range is dinamically set to make room for the tallest bar text.
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
                text="🧁 @lapanquecita",
            ),
        ],
    )

    # We name the resulting figure with the state_id.
    fig.write_image(fr"F:\dsl_CLIMA\projects\submittable\missing persons\plots\mexico\{state_id}.png")


if __name__ == "__main__":
    main(0)