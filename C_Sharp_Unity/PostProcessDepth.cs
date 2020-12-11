using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using JetBrains.Annotations;
using Unity.Collections;

using UnityEngine.Experimental.Rendering;
using Unity.Jobs;
using UnityEngine.SceneManagement;


public class PostProcessDepth : MonoBehaviour
{
    //Use custom script to allow for scene assignment in the inspector
    [Scene]
    public string nextScene;
    public Material mat;

    public RenderTexture depthMotion, final;

    public static string path = "D:\\Other\\UnityOutput\\";


    public int maxWorkers = 8;
    public bool useAutoAnimation = false;

    public static string startTime;
    
    public static Vector2Int dimensions = new Vector2Int(512, 512);


    void Start()
    {
        var dt = DateTime.UtcNow;
        startTime = ((DateTimeOffset) dt).ToUnixTimeSeconds().ToString();
        //startTime = DateTime.Now.GetHashCode().ToString();
        //depthMotion = new RenderTexture(dimensions.x, dimensions.y, 24, RenderTextureFormat.ARGBHalf);
        //final = new RenderTexture(dimensions.x, dimensions.y, 24, RenderTextureFormat.ARGBHalf);
        depthMotion.format = RenderTextureFormat.ARGBHalf;
        final.format = RenderTextureFormat.ARGBHalf;

        sceneName = SceneManager.GetActiveScene().name;
        Physics.autoSimulation = false; //disable automatic physics simulation (so we can tie it to framerate)
        
        GetComponent<Camera>().depthTextureMode = DepthTextureMode.MotionVectors;
        var mode = GetComponent<Camera>().depthTextureMode;
        if (mode == (DepthTextureMode.MotionVectors | DepthTextureMode.Depth))
        {
            Debug.Log("Enabled motion vectors and depth texture.");
        }
        else 
        {
            Debug.Log("ERROR: Couldn't enable motion vectors or depth texture!");
            Debug.Log("Current mode: " + mode);
        }

        System.IO.Directory.CreateDirectory(path + sceneName);

        while (t2d_depthMotion.Count < maxWorkers) //Create textures once, and then overwrite later
        {
            t2d_depthMotion.Add(new Texture2D(dimensions.x, dimensions.y, TextureFormat.RGBAHalf, false));
            t2d_final.Add(new Texture2D(dimensions.x, dimensions.y, TextureFormat.RGBAHalf, false));
        }
        handles = new JobHandle?[maxWorkers];
        for (int i = 0; i < maxWorkers; i++)
        {
            handles[i] = null;
        }

        number = 0;
    }


    //Convert to exr in separate thread to increase performance
    public struct SaveJob : IJob
    {
        public uint number;
        public NativeArray<byte> bytes, bytes2;
        public bool useAutoAnimation;
        public void Execute()
        {
            byte[] data, data2;
            //Export depth and motion vectors as 16 bits per channel EXR to preserve precision
            data = ImageConversion.EncodeArrayToEXR(bytes.ToArray(), GraphicsFormat.R16G16B16A16_SFloat, (uint)dimensions.x, (uint)dimensions.y, 0U, Texture2D.EXRFlags.CompressZIP); 
            data2 = ImageConversion.EncodeArrayToEXR(bytes2.ToArray(), GraphicsFormat.R16G16B16A16_SFloat, (uint)dimensions.x, (uint)dimensions.y, 0U, Texture2D.EXRFlags.CompressZIP);
            string path1, path2;
            path1 = path + sceneName + "\\motVec" + number.ToString("D4") + ".exr";
            path2 = path + sceneName + "\\final" + number.ToString("D4") + ".exr";
            if (useAutoAnimation)
            {
                System.IO.Directory.CreateDirectory(path + sceneName + "\\" + startTime + "\\");
                path1 = path + sceneName + "\\" + startTime + "\\" + startTime + "motVec" + number.ToString("D4") + ".exr";
                path2 = path + sceneName + "\\" + startTime + "\\" + startTime +"final" + number.ToString("D4") + ".exr";
            }
            var writer1 = new BinaryWriter(File.Open(path1, FileMode.Create));
            var writer2 = new BinaryWriter(File.Open(path2, FileMode.Create));

            writer1.Write(data);
            writer2.Write(data2);
            writer1.Close();
            writer2.Close();
        }
    }
    
    public static uint number = 0;
    
    private JobHandle?[] handles;
    List<Texture2D> t2d_depthMotion = new List<Texture2D>();
    List<Texture2D> t2d_final = new List<Texture2D>();
    private static string sceneName = "null";

    

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {

        bool foundSlot = false;

        while (!foundSlot)
        {
            for (int index = 0; index < maxWorkers; index++)
            {

                if (handles[index].HasValue && handles[index].Value.IsCompleted)
                {
                    handles[index].Value.Complete();
                    handles[index] = null; //free slot, indicating that texture is available.
                }

                if (handles[index] == null)
                {
                    foundSlot = true;
                    RenderTexture.active = destination;
                    t2d_depthMotion[index].ReadPixels(new Rect(0, 0, dimensions.x, dimensions.y), 0, 0);


                    RenderTexture.active = final;
                    t2d_final[index].ReadPixels(new Rect(0, 0, dimensions.x, dimensions.y), 0, 0);


                    NativeArray<byte> rawData = t2d_depthMotion[index].GetRawTextureData<byte>();
                    NativeArray<byte> rawData2 = t2d_final[index].GetRawTextureData<byte>();


                    SaveJob saveData = new SaveJob();
                    saveData.useAutoAnimation = useAutoAnimation;
                    saveData.number = number;
                    saveData.bytes = rawData;
                    saveData.bytes2 = rawData2;
                    handles[index] = saveData.Schedule();

                    //render motion vectors
                    Graphics.Blit(source, destination, mat);
                }
            }
        }

        number++;

    }


    //update all animations by one tick 
    public float StepSize;
    public List<Animator> animators;

    public float PhysicsStepSize;

    void Update()
    {
        Physics.Simulate(PhysicsStepSize);
        if (!useAutoAnimation)
        {
            foreach (var animator in animators)
                animator.SetFloat("mot_time", Mathf.Clamp(animator.GetFloat("mot_time") + StepSize, 0.0f, 1.0f));

            if (animators[0].GetFloat("mot_time") >= 0.999) //If animations are finished, go to next scene
            {

                for (int index = 0; index < maxWorkers; index++) //Stop all threads for next scene
                {
                    if (handles[index].HasValue && handles[index].Value.IsCompleted)
                    {

                        handles[index].Value.Complete();
                        handles[index] = null;

                    }

                    SceneManager.LoadScene(nextScene, LoadSceneMode.Single); //Load next scene
                }
            }
        }
            
    }
}
