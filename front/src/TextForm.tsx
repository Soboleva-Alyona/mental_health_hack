import React, {useEffect} from 'react';
import {useState} from 'react'
import * as tf from '@tensorflow/tfjs'
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import axios from "axios";

const TextForm = () => {
    const [post, setPost] = useState("");


    const [ans, setAns] = useState("");


    return (
        <div>
            <form>

                <input
                    type="text"
                    value={post}
                    onChange={(e) => {
                        setPost(e.target.value);
                        setAns(e.target.value);
                        axios.post('http://localhost:12345/predict', {"text": post})
                            .then(res => {
                                console.log(res.data.prediction);
                                // @ts-ignore
                                setAns(res.data.prediction);
                            })
                    }}
                />
            </form>
            <div>
                {ans}
            </div>
        </div>

    );
};

export default TextForm;
