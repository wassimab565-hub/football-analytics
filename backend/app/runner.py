def default_runner(settings):
    def run_job(video, out_dir, progress):
        from ml.pipeline import run

        run(video, settings.detector_weights, settings.pitch_weights, out_dir, progress=progress, render_video=False)

    return run_job
