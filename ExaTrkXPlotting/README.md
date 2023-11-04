# Exa.TrkX Plotting

Common plotting for ML routine.

## Objective

This package is simply provide a unified interface to define a plot, and form useful figures. It also define series of plot for Exa.TrkX particle tracking study routine.

## Structure

This package is separate into two parts:

- ExaTrkXPlotting:
    
    Define unified interface to define a plot. See examples to learn how it works.

- ExaTrkXPlots
    
    Define series of plot for Exa.TrkX particle tracking study routine. Chose some plots you interested in, import it into your script then you are way to go.
    
## Example

It is recommended to see some example before start using this package, they are located in `examples` directory.

Currently, you need to prepare your own data. You can download some of them in [TrackML Chellage](https://www.kaggle.com/c/trackml-particle-identification). For the rest of them, you need to generate by yourself. This will be changed in near feature.

Also note that you may need additional package to read data in example code.

## TODO

- Better API call and external configuration interface.
- More plots!

## Recommendation

You can use [ExaTrkXDataIO](https://github.com/rlf23240/ExaTrkXDataIO) to simplify data handling before you feed into plotter.