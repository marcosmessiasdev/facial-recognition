namespace VisionEngine.Stages;

internal interface IFrameStagePipelineBuilder
{
    IReadOnlyList<IFrameStage> Build();
}

