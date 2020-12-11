using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine;
using Random = UnityEngine.Random;

public class AutomaticAnimation : MonoBehaviour
{
    public Animator animator;

    public AnimationClip animation;

    public int numSequences;

    public float stepSize;

    public GameObject PhysicsGameObjects;

    public Mesh[] meshes;

    public float launchForce;

    public Material[] skyboxes;

    public GameObject ground;

    // Start is called before the first frame update
    void Start()
    {
        animator.SetFloat("mot_time", 0.0f);
        CreateNewAnimation();
        
    }

    // Update is called once per frame
    void Update()
    {
        if (!StepAnimation())
        {
            Debug.Log("Creating new animation");
            animator.SetFloat("mot_time", 0.0f);
            CreateNewAnimation();
        }
    }

    void CreateNewAnimation()
    {
        PostProcessDepth.number = 0;
        RenderSettings.skybox = skyboxes[Random.Range(0, skyboxes.Length)];
        var dt = DateTime.UtcNow;
        PostProcessDepth.startTime =  ((DateTimeOffset)dt).ToUnixTimeSeconds().ToString();
        foreach (var obj in PhysicsGameObjects.GetComponentsInChildren<Transform>())
        {
            if (obj.gameObject.tag == "ignore")
                continue;
            obj.SetPositionAndRotation((Random.insideUnitSphere * 7) + (Vector3.up * 10), Random.rotation); //Random rotation, somewhere inside sphere of radius 7 centered at 0,10,0
            obj.GetComponent<Rigidbody>().AddForce(Random.insideUnitSphere * launchForce); //Apply random force
            var meshComp = obj.GetComponent<MeshFilter>();
            meshComp.sharedMesh = meshes[Random.Range(0, meshes.Length)];
            var colliderComp = obj.GetComponent<MeshCollider>();
            colliderComp.sharedMesh = meshComp.sharedMesh;
            var mat = obj.gameObject.GetComponent<Renderer>().material;
            mat.SetColor("_Color", Random.ColorHSV(0f, 1f, 0.5f, 1.0f, 0.5f, 0.5f));
            mat.SetFloat("_Glossiness", Random.Range(0.0f, 0.7f));
            mat.SetFloat("_Metallic", Random.Range(0.0f, 1.0f));

            var groundMaterial = ground.GetComponent<Renderer>().material;
            groundMaterial.SetColor("_Color", Random.ColorHSV(0f, 1f, 0.1f, 0.9f, 0.2f, 0.7f));

        }

        int numKeys = 10;
        animation.ClearCurves();
        animation.legacy = false;
        Keyframe[][] keys = new Keyframe[7][];
        for (int i = 0; i < 7; i++)
        {
            keys[i] = new Keyframe[numKeys];
        }
        
        for (int i = 0; i < numKeys; i++)
        {
            Vector3 randomPosition = (Random.onUnitSphere * Random.Range(7.5f, 15f)) + (Vector3.up * 10); //Random position 10 units from center (0,10,0)
            keys[0][i] = new Keyframe(i, randomPosition.x);
            keys[1][i] = new Keyframe(i, Mathf.Clamp(randomPosition.y, 10.0f, 12.5f));
            keys[2][i] = new Keyframe(i, randomPosition.z);

            //Vector3 randomRotation = Random.rotationUniform.eulerAngles;
            Quaternion randomRotation = Quaternion.LookRotation(new Vector3(0, 10, 0) - randomPosition);
            randomRotation = Quaternion.RotateTowards(randomRotation, Random.rotationUniform, Random.Range(0.0f, 5f));
            //keys[3][i] = new Keyframe(i, Mathf.Clamp(randomRotation.x, 140, 220));
            //keys[4][i] = new Keyframe(i, randomRotation.y);
            //keys[5][i] = new Keyframe(i, Mathf.Clamp(randomRotation.z, 160, 200));
            keys[3][i] = new Keyframe(i, randomRotation.x);
            keys[4][i] = new Keyframe(i, randomRotation.y);
            keys[5][i] = new Keyframe(i, randomRotation.z);
            keys[6][i] = new Keyframe(i, randomRotation.w);

            if (i > 0 && Random.Range(0.0f, 1.0f) < 0.25f && Random.Range(0.0f, 1.0f) > (i-1)/(float)numKeys) //25% chance to not move, decreases to 2.5% as time goes on
            {
                for (int x = 0; x < 7; x++)
                {
                    keys[x][i] = new Keyframe(i, keys[x][i - 1].value);
                }
            }
        }

        AnimationCurve curveX = new AnimationCurve(keys[0]);
        AnimationCurve curveY = new AnimationCurve(keys[1]);
        AnimationCurve curveZ = new AnimationCurve(keys[2]);
        AnimationCurve rotX = new AnimationCurve(keys[3]);
        AnimationCurve rotY = new AnimationCurve(keys[4]);
        AnimationCurve rotZ = new AnimationCurve(keys[5]);
        AnimationCurve rotW = new AnimationCurve(keys[6]);

        animation.SetCurve("", typeof(Transform), "localPosition.x", curveX);
        animation.SetCurve("", typeof(Transform), "localPosition.y", curveY);
        animation.SetCurve("", typeof(Transform), "localPosition.z", curveZ);
        animation.SetCurve("", typeof(Transform), "localRotation.x", rotX);
        animation.SetCurve("", typeof(Transform), "localRotation.y", rotY);
        animation.SetCurve("", typeof(Transform), "localRotation.z", rotZ);
        animation.SetCurve("", typeof(Transform), "localRotation.w", rotW);
    }

    bool StepAnimation()
    {
        animator.SetFloat("mot_time", Mathf.Clamp(animator.GetFloat("mot_time") + stepSize, 0.0f, 1.0f));
        //Debug.Log("Stepping animation. Progress:" + animator.GetFloat("mot_time"));
        if (animator.GetFloat("mot_time") >= 0.999) //If animation is finished
        {
            return false;
        }

        return true;
    }
}
