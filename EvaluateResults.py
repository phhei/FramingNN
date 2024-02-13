import sys
from collections import defaultdict
from pathlib import Path
from json import load as json_load, dumps as json_dumps
from re import sub as re_sub

import click
from loguru import logger


@click.command(add_help_option=True)
@click.option("--root-path", "-root", "-r",
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
              default=Path(".out"),
              show_default=True,
              help="Root path to search for stats files")
@click.option("--stats-file-pattern", "-pattern", "-p",
              type=str,
              default="test_metrics*.txt",
              show_default=True,
              help="Pattern to search for stats files")
@click.option("--ignore-runs-wo-existing-model-weights", "-ignore", "-i",
              is_flag=True,
              default=False,
              show_default=True,
              help="Ignore runs without existing model weights (RECOMMENDED)")
@click.option("--log-level", "-log", "-l",
              type=click.Choice(
                  ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
                  case_sensitive=False
              ),
              default="INFO",
              show_default=True,
              help="Log level (should be at least SUCCESS for the final output)")
@click.option("--log_single_class_stats", "-class", "-c",
              is_flag=True,
              default=False,
              show_default=True,
              help="Ignore runs without existing model weights (RECOMMENDED)")
@click.option("--latex_friendly", "-latex", "-x",
              is_flag=True,
              default=False,
              show_default=True,
              help="Format the output in a LaTeX-friendly way (copy-paste to a table)")
def evaluate(root_path: Path, stats_file_pattern: str, ignore_runs_wo_existing_model_weights: bool, log_level: str,
             log_single_class_stats: bool, latex_friendly: bool):
    if log_level != "DEBUG":
        logger.remove()
        logger.add(sink=sys.stdout, level=log_level, colorize=True)
        logger.add(
            sink=root_path.joinpath("_EvaluateResults{}{}.log".format(
                "-detailed" if log_single_class_stats else "",
                "-latex" if latex_friendly else ""
            )),
            level=log_level, rotation="10 MB", colorize=False, mode="w"
        )

    stats_files = list(root_path.rglob(pattern=stats_file_pattern))
    logger.info("Found {} stats files", len(stats_files))

    stats = defaultdict(lambda: defaultdict(list))
    for stats_file in stats_files:
        experiment_folder = stats_file.parent
        while experiment_folder.name.startswith("Run"):
            logger.trace("Looking for experiment folder in \"{}\"... climb up", experiment_folder)
            experiment_folder = experiment_folder.parent
        logger.debug("Found experiment folder: \"{}\" for {}", experiment_folder.name, stats_file.stem)
        experiment_name = f"{experiment_folder}=>{stats_file.stem}"

        existing_model_weights = list(stats_file.parent.glob(pattern="model_weights-*"))
        if not ignore_runs_wo_existing_model_weights or len(existing_model_weights) >= 1:
            logger.trace("Found {} existing model weights: {}",
                         len(existing_model_weights), " and ".join([p.stem for p in existing_model_weights]))
            logger.trace("Reading stats file: {} ({} bytes)", stats_file.name, stats_file.stat().st_size)
            try:
                with stats_file.open(mode="r", encoding="utf-8", errors="replace") as f:
                    experiment_scores = json_load(f)
                logger.debug("Read {} scores from {}", len(experiment_scores), stats_file.name)

                for score_name, score_value in experiment_scores.items():
                    if log_single_class_stats or "=>Class" not in score_name:
                        stats[experiment_name][re_sub(string=score_name, pattern="Run\d+", repl="")].append(score_value)
                    else:
                        logger.debug("Ignoring single class score: {}", score_name)
                stats[experiment_name]["_successful_runs"].append(1)
            except IOError:
                logger.opt(exception=True).warning("Could not read stats file: {}", stats_file.absolute())
                stats[experiment_name]["_successful_runs"].append(0)
        else:
            logger.info("No existing model weights found here: {}", stats_file.parent)
            stats[experiment_name]["_successful_runs"].append(0)

    logger.success("Collected all {} stats", sum(map(len, stats.values())))
    logger.debug("Let's build the average stats")
    for experiment_name, scores in stats.items():
        for score_name, score_values in scores.items():
            if score_name == "_successful_runs":
                logger.debug("Percentage of successful runs for {}: {}/{}",
                             experiment_name, sum(score_values), len(score_values))
                stats[experiment_name][score_name] = sum(score_values)
                continue
            if latex_friendly:
                stats[experiment_name][score_name] = \
                    f"${sum(score_values) / len(score_values):.2%}_{{{min(score_values):.1%}}}^{{{max(score_values):.1%}}}$"
            else:
                stats[experiment_name][score_name] = \
                    f"{min(score_values):.1%}--{sum(score_values) / len(score_values):.2%}--{max(score_values):.1%}"

    logger.success(json_dumps(obj=stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    evaluate()
