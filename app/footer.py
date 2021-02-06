# src: https://discuss.streamlit.io/t/st-footer/6447 by chris_klose

import streamlit as st
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, nbsp, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 63px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0,0,0,0),
        width=percent(100),
        height="auto",
        color="black", 
        background_color="#f5f7f8",
        text_align="left",
        opacity=1
    )

    style_hr = styles(
        position="fixed",
        left=0,
        bottom=px(75),
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        #hr(
        #    style=style_hr
        #),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "&nbsp&nbsp Coded in ",
        link("https://www.streamlit.io/", image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(25), height=px(25))),
        " by ",
        link("https://www.linkedin.com/in/poleniuk/", "@PatrykOleniuk"),
        br(),"&nbsp &nbsp",
        link("https://github.com/patryk-oleniuk/face-emotionz-webapp","source code"),
    ]
    layout(*myargs)

