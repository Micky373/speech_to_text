import React from "react";

function Home() {
  return (
    <div className=" h-full w-full border-solid border-red-500 border-2">
      <div className="">
        <h1 className="text-center pt-56 pb-9 font-mono text-3xl">
          INTRODUCTION
        </h1>
        <div className="font-extralight p-10 text-2xl ">
          <p>
            The World Food Program wants to deploy an intelligent form that
            collects nutritional information of food bought and sold at markets
            in three different countries in Africa - Ethiopia and Kenya.
          </p>
          <p>
            The design of this intelligent form requires selected people to
            install an app on their mobile phone, and whenever they buy food,
            they use their voice to activate the app to register the list of
            items they just bought in their own language. The intelligent
            systems in the app are expected to live to transcribe the
            speech-to-text and organize the information in an easy-to-process
            way in a database.
          </p>
          <p>
            It is our obligation to create a deep learning model capable of
            converting speech to text. The model we create should be precise and
            resistant to background noise. This project was created during the
            fourth week of the Machine Learning training session at 10Academy.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Home;
