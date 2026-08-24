import pandas as pd
import wandb

WANDB_USER = "hao5050538-kungliga-tekniska-hogskolan"
PROJECT_NAME = "PathCLIP-PathCap"


def fetch_best_metrics():
    api = wandb.Api()
    entity = WANDB_USER if WANDB_USER else api.default_entity
    print(f" Entity: {entity}")
    print(
        f" Fetching historical data for project '{PROJECT_NAME}' from"
        " WandB...\n"
    )

    try:
        runs = api.runs(f"{entity}/{PROJECT_NAME}")
    except Exception as e:
        print(f"Error finding project: {e}")
        return

    results = []

    for run in runs:
        history = run.history(
            keys=[
                "epoch",
                "val/clip_val_loss",
                "train/loss",
                "val/text_to_image_R@1",
                "val/image_to_text_R@1",
                "val/text_to_image_mean_rank",
            ]
        )

        if (
            "val/clip_val_loss" in history.columns
            and not history["val/clip_val_loss"].dropna().empty
        ):

            # ==========================================================
            # 1. make sure that if train/loss is missing for some epochs, we forward-fill it
            #    so that when validation occurs, it can automatically pick up the "latest train/loss" that just finished running
            history["train/loss"] = history["train/loss"].ffill()

            # 2. find the epoch with the minimum validation loss and extract its metrics
            best_idx = history["val/clip_val_loss"].idxmin()
            best_row = history.loc[best_idx]
            # ==========================================================

            val_loss = best_row["val/clip_val_loss"]
            train_loss = best_row.get("train/loss", float("nan"))
            t2i_r1 = best_row.get("val/text_to_image_R@1", 0)
            i2t_r1 = best_row.get("val/image_to_text_R@1", 0)
            t2i_mr = best_row.get("val/text_to_image_mean_rank", float("nan"))

            # Calculate Gap
            gap = (
                val_loss - train_loss
                if pd.notnull(train_loss)
                else float("nan")
            )

            results.append({
                "Run Name": run.name,
                "Best Epoch": int(best_row["epoch"]),
                "Val Loss ⬇️": round(val_loss, 4),
                "Train Loss": (
                    round(train_loss, 4) if pd.notnull(train_loss) else "N/A"
                ),
                "Loss Gap ⬇️": round(gap, 4) if pd.notnull(gap) else "N/A",
                "T2I R@1 ⬆️": (
                    f"{t2i_r1 * 100:.2f}%" if pd.notnull(t2i_r1) else "N/A"
                ),
                "I2T R@1 ⬆️": (
                    f"{i2t_r1 * 100:.2f}%" if pd.notnull(i2t_r1) else "N/A"
                ),
                "T2I Mean Rank ⬇️": (
                    round(t2i_mr, 2) if pd.notnull(t2i_mr) else "N/A"
                ),
            })

    if not results:
        print("No valid run records found.")
        return

    df = pd.DataFrame(results).sort_values(by="Val Loss ⬇️")

    print("=" * 110)
    print(
        " Automated Comprehensive Evaluation Table (Best Epoch per Run) "
    )
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)


if __name__ == "__main__":
    fetch_best_metrics()