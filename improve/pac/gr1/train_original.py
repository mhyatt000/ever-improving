
def train(
    acc,
    prefetcher,
    # test_prefetcher,
    preprocessor,
    model,
    optimizer,
    scheduler,
    device,
    cfg,
    step,
    writer,
):
    """
    prof = profile(
        schedule = torch.profiler.schedule(
            wait=20,
            warmup=3,
            active=4,
            repeat=1,
        ),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(cfg['save_path']+'prof'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
    prof.start()
    """

    # train_dataset_len = len(prefetcher.loader.dataset)
    # test_dataset_len = len(test_prefetcher.loader.dataset)
    # eval_steps = train_dataset_len // test_dataset_len
    avg_reward = 0.0

    for epoch in range(cfg.num_epochs):

        if epoch % cfg.save_epochs == 0:

            """
            if cfg.evaluate_during_training:
                model.eval()
                avg_reward = ( torch.tensor( evaluate_policy( eva, env, cfg.save_path + "success_rate.txt", cfg.save_path + "result.txt", cfg.ep_len, cfg.num_sequences, acc.num_processes, acc.process_index, eval_dir, debug=cfg.record_evaluation_video,)) .float() .mean() .to(device))
                avg_reward = acc.gather_for_metrics(avg_reward).mean()
            """
            # save_model(acc, model, cfg, epoch, ["model_mae", "model_clip"])
            pass

        loss_keys = ["rgb_static", "rgb_gripper", "action_arm", "action_gripper"]
        loss_val = lambda: torch.tensor(0).float().to(device)
        logger = {k: loss_val() for k in loss_keys}
        eval_log_loss = {k: loss_val() for k in loss_keys}

        cum_load_time = 0
        clock = time()
        batch_idx = 0
        batch, load_time = prefetcher.next()

        # for _ in range(1_000_000): # set as cfg.n_steps
        while batch is not None:
            with acc.accumulate(model):

                model.train()
                optimizer.zero_grad()
                batch["rgb_static"], batch["rgb_gripper"] = preprocessor.rgb_process(
                    batch["rgb_static"], batch["rgb_gripper"], train=True
                )
                obs_mask = batch["mask"][..., 0]

                pred = model(
                    rgb=rgb_static,
                    hand_rgb=rgb_gripper,
                    state={
                        "arm": batch["arm_state"],
                        "gripper": batch["gripper_state"],
                    },
                    language=batch["inst_token"],
                    attention_mask=obs_mask,
                )

                loss = model.loss(pred, batch, obs_mask, cfg.skip_frame)
                total_loss = loss["total"]

                acc.backward(total_loss)
                optimizer.step(optimizer)
                for key in logger:
                    logger[key] += loss[key].detach() / cfg.print_steps
                cum_load_time += load_time / cfg.print_steps

            """
            if batch_idx % eval_steps == 0:  # eval model
                with torch.no_grad():
                    model.eval()
                    batch, _ = test_prefetcher.next_without_none()

                    preprocess the batch
                    get obs mask
                    pred = model()
                    loss = model.loss(pred, batch, obs_mask, cfg.skip_frame)
                    log the info

            """

            if batch_idx % cfg.print_steps == 0 and batch_idx != 0:
                # print to CLI
                # text = "Train Epoch: {} [{}/{} ({:.0f}%)] Reward: {:.5f} FPS:{:.5f} Load Pertentage:{:.5f} LR:{}".format( epoch, batch_idx * cfg.bs_per_gpu * acc.num_processes, train_dataset_len, 100.0 * batch_idx * cfg.bs_per_gpu * acc.num_processes / train_dataset_len, avg_reward, fps, load_pecnt, scheduler.get_last_lr()[0],)

                if acc.is_main_process:
                    # accumulate metrics to the main process
                    # log them to wandb
                    # use the custom wandb logger from improve.sb3.util
                    pass

            batch_idx += 1
            step += 1
            batch, load_time = prefetcher.next()
