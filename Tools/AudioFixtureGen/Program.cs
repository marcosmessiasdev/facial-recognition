using System.Speech.Synthesis;

namespace AudioFixtureGen;

internal static class Program
{
    public static int Main(string[] args)
    {
        string outPath = args.Length > 0
            ? args[0]
            : Path.Combine("E2ETests", "audio", "e2e_fixture_10_words.wav");

        string text = args.Length > 1
            ? string.Join(' ', args.Skip(1))
            : "This offline test audio contains more than ten words for transcription.";

        string fullOut = Path.GetFullPath(outPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullOut)!);

        if (File.Exists(fullOut))
        {
            File.Delete(fullOut);
        }

        using SpeechSynthesizer synth = new();
        synth.Rate = 0;
        synth.Volume = 100;
        synth.SetOutputToWaveFile(fullOut);
        synth.Speak(text);

        Console.WriteLine($"Wrote: {fullOut}");
        Console.WriteLine($"Text: {text}");
        return 0;
    }
}
