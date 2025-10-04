#!/usr/bin/env python3
"""
Main CLI for NLP Lab experiments.

Usage:
    python main.py mask                    # Run mask filling with default text
    python main.py mask -i                 # Run mask filling in interactive mode
    python main.py mask --text "My <mask> text"  # Run mask filling with custom text

    python main.py pos                     # Run POS tagging with default text
    python main.py pos -i                  # Run POS tagging in interactive mode
    python main.py pos --text "Custom text" # Run POS tagging with custom text

    python main.py ner                     # Run NER with default text
    python main.py ner -i                  # Run NER in interactive mode
    python main.py ner --text "Custom text" # Run NER with custom text
"""

import click
from mask import main as mask_main
from pos import main as pos_main
from ner import main as ner_main
from trad import main as trad_main


@click.group()
def cli():
    """NLP Lab experiment runner."""
    pass


@cli.command()
@click.option('-i', '--interactive', is_flag=True,
              help='Run in interactive mode - prompt user for input')
@click.option('--text', type=str, default=None,
              help='Custom text with <mask> token to fill')
def mask(interactive, text):
    """Run mask filling experiment with Catalan RoBERTa model.

    Examples:
        python main.py mask
        python main.py mask -i
        python main.py mask --text "El meu <mask> favorit"
    """
    mask_main(text=text, interactive=interactive)


@cli.command()
@click.option('-i', '--interactive', is_flag=True,
              help='Run in interactive mode - prompt user for input')
@click.option('--text', type=str, default=None,
              help='Custom text to analyze for POS tags')
def pos(interactive, text):
    """Run POS tagging experiment with Catalan RoBERTa model.

    Examples:
        python main.py pos
        python main.py pos -i
        python main.py pos --text "El gat negre dorm"
    """
    pos_main(text=text, interactive=interactive)


@cli.command()
@click.option('-i', '--interactive', is_flag=True,
              help='Run in interactive mode - prompt user for input')
@click.option('--text', type=str, default=None,
              help='Custom text to analyze for named entities')
def ner(interactive, text):
    """Run NER (Named Entity Recognition) experiment with Catalan RoBERTa model.

    Examples:
        python main.py ner
        python main.py ner -i
        python main.py ner --text "Maria viu a Barcelona"
    """
    ner_main(text=text, interactive=interactive)


@cli.command()
@click.option('-i', '--interactive', is_flag=True,
              help='Run in interactive mode - prompt user for input')
@click.option('--text', type=str, default=None,
              help='Text to translate')
@click.option('--combination', type=str, default="fr-ca",
              help='Chose a language combination')
def trad(interactive, text, combination):
    """Run TRAD (translation) experiment with Catalan Aina MT model.

    Examples:
        python main.py trad
        python main.py trad -i
        python main.py trad --text "Bienvenue en Catalogne"
    """
    trad_main(text=text, interactive=interactive, combination=combination)


if __name__ == '__main__':
    cli()
